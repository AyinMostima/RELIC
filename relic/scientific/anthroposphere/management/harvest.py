"""
Harvest Management Module

This module implements crop and forest harvest operations including biomass
removal, residue management, and impacts on carbon and nitrogen cycling.

References:
    - Prince, S.D., et al. (2001). Global primary production: A remote sensing approach. Journal of Biogeography 28, 457-474.
    - Johnson, J.M.F., et al. (2006). Agricultural opportunities to mitigate greenhouse gas emissions. Environmental Pollution 150, 107-124.
    - Lal, R. (2005). World crop residues production and implications of its use as a biofuel. Environment International 31, 575-584.
    - Blanco-Canqui, H. & Lal, R. (2009). Crop residue removal impacts on soil productivity and environmental quality. Critical Reviews in Plant Sciences 28, 139-163.

Mathematical Expression:
    Harvest biomass removal:
    H_biomass = B_total * HI * (1 - R_fraction)
    
    where:
    - H_biomass is harvested biomass [kg DM m-2]
    - B_total is total aboveground biomass [kg DM m-2]
    - HI is harvest index [1]
    - R_fraction is residue retention fraction [1]
    
    Residue return:
    R_C = (B_total - H_biomass) * C_fraction * R_return
    R_N = R_C * (C:N)-1
    
    where R_C, R_N are carbon and nitrogen return rates
"""

import torch
from typing import Any, Dict, List, Mapping, Optional
from enum import Enum
from dataclasses import dataclass

from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin
from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.utils.physical_constants import get_physical_constant
from .event_scheduler import resolve_day_of_year

# Parameter management imports


class CropType(Enum):
    """Crop types with different harvest characteristics"""
    CEREAL_C3 = "cereal_c3"         # Wheat, rice, barley
    CEREAL_C4 = "cereal_c4"         # Maize, sorghum
    LEGUME = "legume"               # Soybean, beans, peas
    ROOT_TUBER = "root_tuber"       # Potato, cassava, sugar beet
    FIBER = "fiber"                 # Cotton
    SUGAR = "sugar"                 # Sugarcane, sugar beet
    OIL_SEED = "oil_seed"           # Sunflower, rapeseed
    FORAGE = "forage"               # Alfalfa, grass
    TREE_FRUIT = "tree_fruit"       # Orchards
    FOREST = "forest"               # Timber harvest


@dataclass
class HarvestEvent:
    """
    Data class for harvest operation events
    
    Attributes:
        crop_type: Type of crop being harvested (selects the carbon content and
            C:N ratio of the harvested material)
        harvest_date: Day of year for harvest
        harvest_index: Fraction of aboveground biomass harvested [1]
        residue_retention: Fraction of the non-harvested aboveground biomass
            retained as residue [1]
        quality_factor: Quality factor affecting harvest efficiency [1]

    Every field enters the partition. The field defaults describe a generic C3
    cereal harvest and are construction values of a user-specified event, not
    model parameters.
    """
    crop_type: CropType = CropType.CEREAL_C3
    harvest_date: int = 270  # Day of year (late September)
    harvest_index: float = 0.45  # Fraction
    residue_retention: float = 0.6  # Fraction
    quality_factor: float = 1.0  # 1.0 = normal quality


class HarvestManagement(NamespaceDefaultMixin, BiologicalProcessFunctor):
    """
    Harvest Management Functor
    
    Implements crop and forest harvest operations, biomass removal,
    and residue management with effects on carbon and nitrogen cycling.
    
    Parameters (anthroposphere.management.harvest.*):
        harvest_efficiency: Mechanical harvest efficiency [1]
        root_mortality_rate: Root mortality fraction of a harvest [1]
        <crop>_c_content, <crop>_cn_ratio: Carbon content [kg C kg-1 DM] and C:N
            ratio [kg C kg-1 N] of the harvested material per crop type
        default_harvest_index, residue_removal_rate, residue_retention: Hodge
            weights of the partition audit [1]
        schedule_window_days, minimum_event_interval_days, default_schedule.*:
            scheduling conventions and the default scenario
    """
    
    parameter_lookup_namespace = "anthroposphere.management.harvest"
    parameter_lookup_source_order = ("config", "manager")
    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    EXTRA_STATE_FORMAT = "relic.harvest.runtime_state"
    EXTRA_STATE_VERSION = 2
    PHYSICAL_CONSTANT_KEYS = (
        "relic.physical.constants.time.seconds_per_day",
    )

    def __init__(self, name: str = "harvest_management", config: Optional[Dict] = None):
        """
        Initialize harvest management functor
        
        Args:
            name: Functor name
            config: Configuration dictionary
        """
        # Set up metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Harvest management and biomass removal for agriculture and forestry",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "Canonical IPHS: x_dot = M_h(u_h,theta) mu_h + G_h u_harvest; "
                "H_biomass = B_total * HI * (1 - R_fraction); "
                "R_C = (B_total - H_biomass) * C_fraction * R_return"
            ),
            references=[
                "Prince et al. (2001) DOI:10.1046/j.1365-2699.2001.00567.x",
                "Johnson et al. (2006) DOI:10.2134/agronj2005.0179",
                "Lal (2005) DOI:10.1016/j.envint.2004.09.005",
                "Blanco-Canqui & Lal (2009) DOI:10.1080/07352680902776507"
            ],
            parameters={
                "anthroposphere.management.harvest.harvest_efficiency": (
                    "Mechanical harvest efficiency [dimensionless]"
                ),
                "anthroposphere.management.harvest.default_harvest_index": (
                    "Harvest index weight of the harvest partition Hodge contract [dimensionless]"
                ),
                "anthroposphere.management.harvest.residue_retention": (
                    "Fraction of residue left on field [dimensionless]"
                ),
                "anthroposphere.management.harvest.residue_removal_rate": (
                    "Fraction of residue removed during harvest operations [dimensionless]"
                ),
                "anthroposphere.management.harvest.root_mortality_rate": (
                    "Root mortality fraction triggered by harvest disturbance [dimensionless]"
                ),
                "anthroposphere.management.harvest.*_c_content": (
                    "Carbon content of the harvested dry matter per crop type (cereal C3, "
                    "cereal C4, legume, root and tuber, forage, forest) [kg C kg-1 DM]"
                ),
                "anthroposphere.management.harvest.*_cn_ratio": (
                    "C:N ratio of the harvested material per crop type (cereal C3, cereal C4, "
                    "legume, root and tuber, forage, forest) [kg C kg-1 N]"
                ),
                "anthroposphere.management.harvest.schedule_window_days": (
                    "Half-width of the calendar window around a scheduled harvest day [day]"
                ),
                "anthroposphere.management.harvest.minimum_event_interval_days": (
                    "Minimum elapsed time after an accepted harvest before the next one [day]"
                ),
                "anthroposphere.management.harvest.default_schedule.*": (
                    "Date [day], harvest index [dimensionless] and residue retention "
                    "[dimensionless] of the default cereal harvest and forage cuts "
                    "[units per key]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "General relative tolerance of the conservation checks [dimensionless]"
                ),
            },
            input_variables=[
                "aboveground_biomass",
                "belowground_biomass",
                "residue_cover",
                "day_of_year",
            ],
            output_variables=[
                "harvest_flux",
                "harvest_biomass_carbon_input",
                "harvest_carbon_export_flux",
                "residue_c_flux",
                "harvest_biomass_nitrogen_input",
                "harvest_nitrogen_export_flux",
                "residue_n_flux",
                "delta_residue_cover",
                "diagnostics.harvest.events_today",
                "diagnostics.harvest.biomass_partition_residual",
                "diagnostics.harvest.carbon_ledger_residual",
                "diagnostics.harvest.nitrogen_ledger_residual",
                "diagnostics.harvest_management.free_energy",
                "diagnostics.harvest_management.chemical_potential_norm",
                "diagnostics.harvest_management.canonical_iphs_runtime",
            ],
            conservation_laws=["mass"],  # Carbon and nitrogen mass balance
            spatial_dimensions=2,
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=["biosphere", "pedosphere"]
        )
        
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata
        )
        configured_calendar = (config or {}).get("calendar_type")
        self._configured_calendar_type = (
            None
            if configured_calendar is None
            else str(configured_calendar)
        )
        
        # Initialize harvest schedule
        self.harvest_events: List[HarvestEvent] = []
        self._initialize_default_schedule()
        
        # Harvest tracking
        self.last_harvest_time_seconds: Optional[float] = None
        self.cumulative_biomass_removed = 0.0  # kg DM/m^2
        self.cumulative_c_removed = 0.0        # kg C/m^2
        self.cumulative_residue_returned = 0.0  # kg DM/m^2
        
        # Yield tracking
        self.annual_yield = 0.0  # kg DM/m^2/yr
        self.harvest_count = 0
        self.runtime_path = "canonical_iphs_harvest_biomass_residue_port"

    def get_extra_state(self) -> Dict[str, Any]:
        """Return replay-critical harvest accounting for step rollback.

        References:
          - PyTorch Module extra-state contract.
          - Revised RELIC foundations, accepted-step transaction contract.
        Mathematical expression:
          z_hidden' = z_hidden + Delta z_harvest; rollback restores z_hidden.
        Parameter meanings:
          z_hidden contains the last accepted event time [s], cumulative
          biomass/carbon/residue, annual yield, and accepted event count.
        """

        return {
            "format": self.EXTRA_STATE_FORMAT,
            "version": self.EXTRA_STATE_VERSION,
            "last_harvest_time_seconds": self.last_harvest_time_seconds,
            "cumulative_biomass_removed": self.cumulative_biomass_removed,
            "cumulative_c_removed": self.cumulative_c_removed,
            "cumulative_residue_returned": self.cumulative_residue_returned,
            "annual_yield": self.annual_yield,
            "harvest_count": self.harvest_count,
        }

    def set_extra_state(self, state: Any) -> None:
        """Restore replay-critical harvest accounting after strict validation."""

        if not isinstance(state, Mapping):
            raise TypeError("Harvest extra state must be a mapping.")
        expected_keys = {
            "format",
            "version",
            "last_harvest_time_seconds",
            "cumulative_biomass_removed",
            "cumulative_c_removed",
            "cumulative_residue_returned",
            "annual_yield",
            "harvest_count",
        }
        if set(state) != expected_keys:
            raise ValueError("Harvest extra-state keys do not match the required schema.")
        if state["format"] != self.EXTRA_STATE_FORMAT:
            raise ValueError("Harvest extra-state format is unsupported.")
        if state["version"] != self.EXTRA_STATE_VERSION:
            raise ValueError("Harvest extra-state version is unsupported.")

        last_time = state["last_harvest_time_seconds"]
        count = state["harvest_count"]
        restored_last_time: Optional[float] = None
        if last_time is not None:
            if isinstance(last_time, bool) or not isinstance(last_time, (int, float)):
                raise TypeError(
                    "Harvest last_harvest_time_seconds must be numeric or None."
                )
            last_time_tensor = torch.as_tensor(last_time, dtype=torch.float64)
            if (
                not bool(torch.isfinite(last_time_tensor))
                or bool(torch.lt(last_time_tensor, 0.0))
            ):
                raise ValueError(
                    "Harvest last_harvest_time_seconds must be finite and "
                    "non-negative."
                )
            restored_last_time = self._scalar_from_value(last_time_tensor)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Harvest harvest_count must be a non-negative integer.")
        restored_scalars: Dict[str, float] = {}
        for key in (
            "cumulative_biomass_removed",
            "cumulative_c_removed",
            "cumulative_residue_returned",
            "annual_yield",
        ):
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Harvest {key} must be numeric.")
            numeric = float(value)
            finite = torch.isfinite(torch.as_tensor(numeric, dtype=torch.float64))
            if not bool(finite) or numeric < 0.0:
                raise ValueError(f"Harvest {key} must be finite and non-negative.")
            restored_scalars[key] = numeric

        self.last_harvest_time_seconds = restored_last_time
        self.harvest_count = count
        self.cumulative_biomass_removed = restored_scalars[
            "cumulative_biomass_removed"
        ]
        self.cumulative_c_removed = restored_scalars["cumulative_c_removed"]
        self.cumulative_residue_returned = restored_scalars[
            "cumulative_residue_returned"
        ]
        self.annual_yield = restored_scalars["annual_yield"]

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Resolve structural audit size for harvest partition channels."""

        value = 5 if n_dofs is None else int(n_dofs)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare cochain spaces for the harvest biomass-residue port map.

        References:
          - Lal (2005), crop residue production and use.
          - Johnson et al. (2006), agricultural residue greenhouse gas mitigation.
          - Ramirez and Le Gorrec (2022), irreversible port-Hamiltonian systems.
        Mathematical expression:
          x_dot = M_h(u_h,theta) mu_h + G_h u_harvest, with
          H_biomass = B_total * HI * (1 - R_fraction).
        Parameter meanings:
          x is harvestable biomass [kg DM m^-2], M_h is the harvest partition
          mobility [s^-1], mu_h is the storage effort, and u_harvest is the
          management harvest control port.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_harvest",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="kg m^-2",
                extensivity="density",
                physical_domain="biosphere",
                semantic_name="harvest_biomass_pool",
                orientation_convention="positive_crop_storage",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_harvest",
                degree=0,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m^-2 s^-1",
                extensivity="density",
                physical_domain="pedosphere",
                semantic_name="harvest_residue_flux",
                orientation_convention="positive_residue_return",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_harvest",
                degree=0,
                primality="dual",
                n_dofs=1,
                units="1",
                extensivity="intensive",
                physical_domain="anthroposphere",
                semantic_name="harvest_management_port",
                orientation_convention="positive_harvest_control",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare the ParameterManager keys (and full-key families) this Functor reads;
        the same entries as metadata.parameters and the IPHS certificate."""

        return (
            "anthroposphere.management.harvest.harvest_efficiency",
            "anthroposphere.management.harvest.default_harvest_index",
            "anthroposphere.management.harvest.residue_retention",
            "anthroposphere.management.harvest.residue_removal_rate",
            "anthroposphere.management.harvest.root_mortality_rate",
            "anthroposphere.management.harvest.*_c_content",
            "anthroposphere.management.harvest.*_cn_ratio",
            "anthroposphere.management.harvest.schedule_window_days",
            "anthroposphere.management.harvest.minimum_event_interval_days",
            "anthroposphere.management.harvest.default_schedule.*",
            "validation.conservation.conservation_tolerance",
        )

    def physical_constant_keys(self) -> tuple[str, ...]:
        """Return physical constants kept outside the domain parameter map."""

        return self.PHYSICAL_CONSTANT_KEYS

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive harvest partition mobility Hodge contract.

        References:
          - Blanco-Canqui and Lal (2009), residue removal impacts.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical expression:
          H_harvest = diag(w_i / sum_j w_j), w_i >= 0.
        Parameter meanings:
          w_i are ParameterManager-backed harvest efficiency, harvest index,
          residue removal, root mortality, and residue retention weights.
        """
        # References: Blanco-Canqui and Lal (2009); van der Schaft and Maschke (1995).
        # Mathematical expression: H_h = diag(w_i / sum_j w_j), with w_i >= 0.
        # Parameter meanings: w_i are harvest partition weights and n_dofs is support size.
        # Units: weights and matrix entries [1]; domain [kg m-2]; codomain [kg m-2 s-1].
        # Domain/constraints: n_dofs >= 1; weights are finite, non-negative, and have positive sum.
        # Sign convention: positive diagonal mobility maps storage toward positive export/return flows.

        n_dofs = self._structural_n_dofs(n_dofs)
        keys = (
            "harvest_efficiency",
            "default_harvest_index",
            "residue_removal_rate",
            "root_mortality_rate",
            "residue_retention",
        )
        values = []
        for key in keys:
            value = self.get_parameter_value(
                key,
                namespace="anthroposphere.management.harvest",
                required=True,
            )
            values.append(abs(self._scalar_from_value(value)))
        if n_dofs > len(values):
            raise ValueError(
                f"The harvest Hodge contract has {len(values)} weighted channels; "
                f"{n_dofs} channels were requested."
            )
        weights = torch.as_tensor(
            values[:n_dofs],
            device=getattr(self, "device", torch.device("cpu")),
            dtype=torch.float32,
        )
        total = torch.sum(weights)
        if total <= 0:
            raise ValueError("Harvest partition weights require positive support.")
        matrix = torch.diag(weights / total)
        domain = CochainSpaceSpec(
            complex_id="anthroposphere_harvest",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="kg m^-2",
            extensivity="density",
            physical_domain="biosphere",
            semantic_name="harvest_biomass_pool",
            orientation_convention="positive_crop_storage",
            boundary_role="interior",
        )
        codomain = CochainSpaceSpec(
            complex_id="anthroposphere_harvest",
            degree=0,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m^-2 s^-1",
            extensivity="density",
            physical_domain="pedosphere",
            semantic_name="harvest_residue_flux",
            orientation_convention="positive_residue_return",
            boundary_role="interior",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_partition_mobility_hodge",
            domain=domain,
            codomain=codomain,
            matrix=matrix,
            parameter_keys=tuple(f"anthroposphere.management.harvest.{key}" for key in keys),
            metric_keys=("validation.conservation.conservation_tolerance",),
            material_law="harvest_biomass_residue_partition_mobility",
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
        """Return the harvest storage potential used by the IPHS runtime."""

        _ = state
        return {
            "harvest_free_energy": "G_h(x)=0.5*x_h.T*H_harvest*x_h",
            "harvest_chemical_potential": "mu_h=dG_h/dx_h=H_harvest*x_h",
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the non-negative harvest partition entropy audit term."""

        _ = state
        return {
            "harvest_partition_entropy": "sigma = z.T H_harvest z >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the declared harvest partition metadata."""
        # References: Grmela and Ottinger (1997); van der Schaft and Maschke (1995).
        # Mathematical expression: x_dot = M_h(u,theta) mu_h + G_h u_harvest.
        # Parameter meanings: x is biomass storage, mu effort, and u management control.
        # Units: x [kg m-2]; x_dot and flow ports [kg m-2 s-1]; effort [1].
        # Domain/constraints: state variables and ports use the declared harvest cochain spaces.
        # Sign convention: positive harvest port flow is outward; positive residue return is inward.

        _ = state, complex_bundle
        return {
            "state_vars": (
                "aboveground_biomass",
                "belowground_biomass",
                "harvest_flux",
                "residue_c_flux",
                "root_c_flux",
            ),
            "reaction_map": "x_dot = M_h(u_h,theta) mu_h + G_h u_harvest",
            "potential_map": "mu_h = dG_h/dx_h = H_harvest x_h",
            "mass_conserving_partition": True,
            "ports": self.ports(),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive harvest partition mobility operator."""
        # References: Blanco-Canqui and Lal (2009); Grmela and Ottinger (1997).
        # Mathematical expression: J = 0 and M = H_h, with z.T M z >= 0.
        # Parameter meanings: M is harvest mobility and z is a compatible effort vector.
        # Units: M entries [1]; connected state/flow spaces [kg m-2] and [kg m-2 s-1].
        # Domain/constraints: H_h is square, finite, normalized, and positive semidefinite.
        # Sign convention: non-negative mobility drives declared export/return orientations.

        _ = state, complex_bundle
        hodge = self.build_hodge_contract(strict=True)
        matrix = hodge.matrix
        return {
            "state_vars": tuple(f"harvest_biomass_pool[{idx}]" for idx in range(int(matrix.shape[0]))),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the harvest biomass export management port."""

        effort_space = CochainSpaceSpec(
            complex_id="anthroposphere_harvest",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="1",
            extensivity="intensive",
            physical_domain="anthroposphere",
            semantic_name="harvest_management_effort",
            orientation_convention="positive_harvest_control",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="anthroposphere_harvest",
            degree=0,
            primality="dual",
            n_dofs=1,
            units="kg m^-2 s^-1",
            extensivity="density",
            physical_domain="biosphere",
            semantic_name="harvest_biomass_export_flux",
            orientation_convention="positive_biomass_export",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="harvest_biomass_residue_port",
                effort_var="harvest_management_effort",
                flow_var="harvest_flux",
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
        """Run the canonical IPHS harvest biomass-residue runtime."""

        return self.forward(dict(state), time=time, dt=dt)

    def _seconds_per_day_value(self) -> float:
        """Resolve the physical day length through the physical constants SSOT."""

        return self._scalar_from_value(
            get_physical_constant("seconds_per_day")
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
        outputs["diagnostics.harvest_management.canonical_iphs_runtime"] = torch.ones(
            (), device=device, dtype=dtype
        )
        return outputs

    @staticmethod
    def _scalar_from_value(value: Any) -> float:
        """Convert scalar tensor or numeric value to float without direct scalar extraction sync calls."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Expected scalar tensor value.")
            detached = value.detach().reshape(-1)
            values = detached.tolist()
            if not values:
                raise ValueError("Expected non-empty scalar tensor value.")
            return float(values[0])
        return float(value)
    
    def _initialize_default_schedule(self) -> None:
        """Initialize the default harvest schedule from the parameter database.

        The default scenario is one C3 cereal harvest and two forage cuts; their
        dates, harvest indices and residue retention fractions are the scenario
        conventions anthroposphere.management.harvest.default_schedule.*.
        """
        scenario = (
            ("cereal", CropType.CEREAL_C3),
            ("forage_first_cut", CropType.FORAGE),
            ("forage_second_cut", CropType.FORAGE),
        )
        for name, crop_type in scenario:
            def _field(field: str) -> float:
                return self.require_scalar(
                    f"default_schedule.{name}.{field}",
                    namespace="anthroposphere.management.harvest",
                )

            date_value = _field("date")
            if date_value != int(date_value):
                raise ValueError(
                    f"default_schedule.{name}.date must be an integer day, got {date_value}"
                )
            self.harvest_events.append(
                HarvestEvent(
                    crop_type=crop_type,
                    harvest_date=int(date_value),
                    harvest_index=_field("harvest_index"),
                    residue_retention=_field("residue_retention"),
                )
            )
    
    def add_harvest_event(self, event: HarvestEvent) -> None:
        """
        Add a harvest operation event to schedule
        
        Args:
            event: HarvestEvent defining harvest details
        """
        self.harvest_events.append(event)
    
    def _get_crop_characteristics(self, crop_type: CropType) -> Dict[str, float]:
        """
        Get standard characteristics for crop types
        
        Args:
            crop_type: Type of crop
            
        Returns:
            Dictionary with the carbon content and the C:N ratio
        """
        # References: Prince et al. (2001); Lal (2005); Johnson et al. (2006).
        # Mathematical expression: theta_crop = lookup(crop_type, parameter database).
        # Parameter meanings: theta_crop contains the carbon fraction f_C and the C:N ratio
        #   of the harvested material of the crop type.
        # Units: C content [kg C kg-1 DM]; C:N [kg C kg-1 N].
        # Domain/constraints: crop_type is a CropType with parameters in the database; a
        #   crop type without them raises (no generic default crop values).
        # Sign convention: all factors are non-negative magnitudes; this lookup assigns no flux direction.
        def harvest_param(name: str, units: str = "dimensionless") -> float:
            value = self.get_parameter_value(
                name,
                namespace="anthroposphere.management.harvest",
                units=units,
                required=True,
            )
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Expected scalar for '{name}'")
                return self._scalar_from_value(value)
            return float(value)

        parameter_prefix = {
            CropType.CEREAL_C3: "cereal_c3",
            CropType.CEREAL_C4: "cereal_c4",
            CropType.LEGUME: "legume",
            CropType.ROOT_TUBER: "root_tuber",
            CropType.FORAGE: "forage",
            CropType.FOREST: "forest",
        }
        if not isinstance(crop_type, CropType):
            raise TypeError(f"crop_type must be a CropType, got {type(crop_type).__name__}")
        if crop_type not in parameter_prefix:
            raise ValueError(
                f"Crop type '{crop_type.value}' has no carbon content and C:N parameters "
                "(anthroposphere.management.harvest.<crop>_c_content and <crop>_cn_ratio)."
            )
        prefix = parameter_prefix[crop_type]
        return {
            "c_content": harvest_param(f"{prefix}_c_content", units="kg C kg-1 DM"),
            "cn_ratio": harvest_param(f"{prefix}_cn_ratio", units="kg C kg-1 N"),
        }
    
    def _check_harvest_schedule(
        self,
        day_of_year: int,
        *,
        calendar_type: str,
        current_time_seconds: Any,
    ) -> List[HarvestEvent]:
        """Check which harvest events should occur today.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, time-window
            and calendar contract.
        Mathematical expression:
            d_C(a,b) = min(|a-b|, Y(C)-|a-b|); active iff d_C <= W
            and t_now - t_last > T_min seconds_per_day.
        Parameter meanings:
            a is current day [day], b is harvest day [day], Y(C) is calendar
            year length [day], t_now and t_last are accepted elapsed simulation
            times [s], W the schedule window and T_min the minimum interval
            after an accepted harvest [day] (scheduling conventions).

        Args:
            day_of_year: Current day of the configured simulator year
            calendar_type: Existing simulator calendar identifier
            current_time_seconds: Accepted elapsed simulation time [s]

        Returns:
            List of harvest events to execute today
        """
        # References: CF Conventions 1.8; revised RELIC calendar contract.
        # Mathematical expression: d_C=min(|a-b|,Y(C)-|a-b|); active iff d_C<=W and
        #   dt_last > T_min seconds_per_day.
        # Parameter meanings: a is current day, b event day, Y year length, dt_last elapsed
        #   time, W schedule_window_days and T_min minimum_event_interval_days (RELIC
        #   scheduling conventions without a literature value).
        # Units: calendar distances, W and T_min [day]; current and last accepted times [s].
        # Domain/constraints: days are calendar-valid, time is finite/non-decreasing, and events are typed.
        # Sign convention: elapsed time and day distance are non-negative; selection has no flux sign.
        day_of_year = CalendarHandler.validate_simulator_day(
            day_of_year,
            calendar_type,
        )
        CalendarHandler.day_of_year_from_elapsed_seconds(
            current_time_seconds,
            calendar_type,
        )
        current_time = torch.as_tensor(current_time_seconds).detach().to(
            dtype=torch.float64
        )
        window_days = self.require_scalar(
            "schedule_window_days",
            namespace="anthroposphere.management.harvest",
        )
        minimum_interval_days = self.require_scalar(
            "minimum_event_interval_days",
            namespace="anthroposphere.management.harvest",
        )
        duplicate_window = torch.as_tensor(
            minimum_interval_days * self._seconds_per_day_value(),
            dtype=current_time.dtype,
            device=current_time.device,
        )
        recently_accepted = False
        if self.last_harvest_time_seconds is not None:
            last_time = torch.as_tensor(
                self.last_harvest_time_seconds,
                dtype=current_time.dtype,
                device=current_time.device,
            )
            elapsed_since_acceptance = current_time - last_time
            if bool(torch.lt(elapsed_since_acceptance, 0.0)):
                raise ValueError(
                    "Harvest simulation time precedes the last accepted event."
                )
            recently_accepted = bool(
                torch.le(elapsed_since_acceptance, duplicate_window)
            )

        events_today = []
        days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)

        for event in self.harvest_events:
            harvest_day = CalendarHandler.validate_simulator_day(
                event.harvest_date,
                calendar_type,
            )
            direct_distance = abs(day_of_year - harvest_day)
            event_distance = min(
                direct_distance,
                days_per_year - direct_distance,
            )
            if event_distance <= window_days and not recently_accepted:
                events_today.append(event)

        return events_today
    
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        time: float,
        dt: float,
        calendar_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate harvest operations and residue return.

        References:
            Johnson et al. (2006); Lal (2005); Blanco-Canqui and Lal (2009);
            CF-1.8 calendar coordinates.
        Mathematical expression:
            H = B HI (1 - R_fraction),
            R_C = (B - H) f_C R_return, and event day
            d = floor(t / seconds_per_day) mod Y(C) + 1.
        Parameter meanings:
            B and H are biomass [kg DM m-2], HI and R_fraction are fractions
            [1], R_C is residue carbon [kg C m-2], and C is the explicit
            simulator calendar.
        """
        # References: Johnson et al. (2006); Lal (2005); CF-1.8.
        # Mathematical expression: H = B * HI * (1 - R_fraction).
        # Parameter meanings: B and H are [kg DM m-2], dt is [s], and C is the calendar.
        # Units: biomass [kg DM m-2]; C/N flux [kg element m-2 s-1]; dt [s].
        # Domain/constraints: state tensors are finite/non-negative, fractions in [0,1], and dt > 0.
        # Sign convention: product/export is outward-positive; residue/root return is inward-positive.
        local_state = dict(state)

        # Get state variables
        # The biomass pools [kg m-2] are declared inputs with no ParameterManager
        # default: a missing value raises, as in the governed call. Each pool is
        # checked against its physical domain, the contract bounds (non-negative
        # mass); a value outside raises (no clipping).
        aboveground_biomass = self._get_state_variable(
            local_state,
            "aboveground_biomass",
            units="kg m-2",
            validate_range=(0.0, float("inf"))
        )
        belowground_biomass = self._get_state_variable(
            local_state,
            "belowground_biomass",
            units="kg m-2",
            validate_range=(0.0, float("inf"))
        )
        target_dtype = torch.get_default_dtype()
        aboveground_biomass = aboveground_biomass.to(dtype=target_dtype)
        belowground_biomass = belowground_biomass.to(dtype=target_dtype)
        day_input = (
            state["day_of_year"]
            if "day_of_year" in state
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
        # The storage potential and effort audit depends on the stocks alone.
        audit_outputs = self._storage_potential(aboveground_biomass, belowground_biomass)

        # References: Johnson et al. (2006), DOI:10.2134/agronj2005.0179;
        # Lal (2005), DOI:10.1016/j.envint.2004.09.005.
        # Mathematical expressions:
        #   B_in = B_product + B_residue + B_root
        #   C_in = C_product + C_residue
        #   N_in = N_product + N_residue
        #   F_q = Delta q / Delta t_event
        # Parameter meanings:
        #   B is dry matter [kg DM m-2], C and N are elemental amounts
        #   [kg C m-2, kg N m-2], and Delta t_event is one physical day [s].
        zero = torch.zeros_like(aboveground_biomass, dtype=target_dtype)
        harvest_amount = zero.clone()
        carbon_input_amount = zero.clone()
        carbon_export_amount = zero.clone()
        residue_carbon_amount = zero.clone()
        nitrogen_input_amount = zero.clone()
        nitrogen_export_amount = zero.clone()
        residue_nitrogen_amount = zero.clone()
        residue_cover_increment = zero.clone()
        remaining_aboveground = aboveground_biomass.clone()
        remaining_belowground = belowground_biomass.clone()
        # residue_cover [1] is a declared input that no RELIC Functor writes (a
        # forcing contract advanced through delta_residue_cover); a missing cover
        # raises instead of assuming bare soil.
        if local_state.get("residue_cover") is None:
            raise ValueError(
                f"{self.name} needs 'residue_cover' [1] as state or forcing; "
                "no default is applied."
            )
        remaining_residue_cover = torch.broadcast_to(
            torch.as_tensor(
                local_state["residue_cover"],
                device=zero.device,
                dtype=zero.dtype,
            ),
            zero.shape,
        ).clone()
        staged_cumulative_biomass = 0.0
        staged_cumulative_carbon = 0.0
        staged_cumulative_residue = 0.0
        staged_annual_yield = 0.0

        events_today = self._check_harvest_schedule(
            day_of_year,
            calendar_type=calendar_type,
            current_time_seconds=time,
        )

        for event in events_today:
            event_state = dict(local_state)
            event_state["aboveground_biomass"] = remaining_aboveground
            event_state["belowground_biomass"] = remaining_belowground
            event_state["residue_cover"] = remaining_residue_cover
            event_rates = self._canonical_iphs_harvest_partition(
                event_state,
                event=event,
            )
            characteristics = self._get_crop_characteristics(event.crop_type)
            c_content = torch.as_tensor(
                characteristics["c_content"],
                device=zero.device,
                dtype=zero.dtype,
            )
            cn_ratio = torch.as_tensor(
                characteristics["cn_ratio"],
                device=zero.device,
                dtype=zero.dtype,
            )
            if torch.any(cn_ratio <= 0.0):
                raise ValueError("Harvest residue C:N ratio must be positive.")

            harvested = torch.clamp(event_rates["grain_yield_rate"], min=0.0)
            residue_dry_matter = torch.clamp(
                event_rates["residue_generation_rate"],
                min=0.0,
            )
            root_dry_matter = torch.clamp(
                event_rates["root_mortality_flux"],
                min=0.0,
            )
            aboveground_transfer = harvested + residue_dry_matter
            if torch.any(aboveground_transfer > remaining_aboveground):
                raise ValueError(
                    "Harvest aboveground partition exceeds available biomass."
                )
            if torch.any(root_dry_matter > remaining_belowground):
                raise ValueError(
                    "Harvest root partition exceeds available belowground biomass."
                )

            harvest_carbon = harvested * c_content
            returned_carbon = (residue_dry_matter + root_dry_matter) * c_content
            harvest_nitrogen = harvest_carbon / cn_ratio
            returned_nitrogen = returned_carbon / cn_ratio
            total_carbon = harvest_carbon + returned_carbon
            total_nitrogen = harvest_nitrogen + returned_nitrogen

            harvest_amount = harvest_amount + harvested
            carbon_input_amount = carbon_input_amount + total_carbon
            carbon_export_amount = carbon_export_amount + harvest_carbon
            residue_carbon_amount = residue_carbon_amount + returned_carbon
            nitrogen_input_amount = nitrogen_input_amount + total_nitrogen
            nitrogen_export_amount = nitrogen_export_amount + harvest_nitrogen
            residue_nitrogen_amount = residue_nitrogen_amount + returned_nitrogen

            next_residue_cover = torch.clamp(
                remaining_residue_cover
                + torch.clamp(event_rates["delta_residue_cover"], min=0.0),
                min=0.0,
                max=1.0,
            )
            residue_cover_increment = (
                residue_cover_increment
                + next_residue_cover
                - remaining_residue_cover
            )
            remaining_residue_cover = next_residue_cover
            remaining_aboveground = torch.clamp(
                remaining_aboveground - aboveground_transfer,
                min=0.0,
            )
            remaining_belowground = torch.clamp(
                remaining_belowground - root_dry_matter,
                min=0.0,
            )

            with torch.no_grad():
                staged_cumulative_biomass += self._scalar_from_value(
                    harvested.detach().mean()
                )
                staged_cumulative_carbon += self._scalar_from_value(
                    harvest_carbon.detach().mean()
                )
                staged_cumulative_residue += self._scalar_from_value(
                    (residue_dry_matter + root_dry_matter).detach().mean()
                )
                staged_annual_yield += self._scalar_from_value(
                    harvested.detach().mean()
                )

        seconds_per_day = torch.as_tensor(
            self._seconds_per_day_value(),
            device=zero.device,
            dtype=zero.dtype,
        )
        harvest_flux = harvest_amount / seconds_per_day
        carbon_input_flux = carbon_input_amount / seconds_per_day
        carbon_export_flux = carbon_export_amount / seconds_per_day
        residue_c_flux = residue_carbon_amount / seconds_per_day
        nitrogen_input_flux = nitrogen_input_amount / seconds_per_day
        nitrogen_export_flux = nitrogen_export_amount / seconds_per_day
        residue_n_flux = residue_nitrogen_amount / seconds_per_day
        residue_cover_rate = residue_cover_increment / seconds_per_day

        biomass_removed = aboveground_biomass - remaining_aboveground
        biomass_partition_residual = self._check_conservation(
            total_aboveground=aboveground_biomass,
            removed_biomass=biomass_removed,
            residue_biomass=remaining_aboveground,
        )
        carbon_ledger_residual = torch.abs(
            carbon_input_flux - carbon_export_flux - residue_c_flux
        )
        nitrogen_ledger_residual = torch.abs(
            nitrogen_input_flux - nitrogen_export_flux - residue_n_flux
        )
        checked_tensors = (
            harvest_flux,
            carbon_input_flux,
            carbon_export_flux,
            residue_c_flux,
            nitrogen_input_flux,
            nitrogen_export_flux,
            residue_n_flux,
            residue_cover_rate,
            biomass_partition_residual,
            carbon_ledger_residual,
            nitrogen_ledger_residual,
        )
        if any(
            value.numel() == 0 or not bool(torch.all(torch.isfinite(value)))
            for value in checked_tensors
        ):
            raise ValueError("Harvest runtime produced a non-finite or empty result.")

        if events_today:
            self.cumulative_biomass_removed += staged_cumulative_biomass
            self.cumulative_c_removed += staged_cumulative_carbon
            self.cumulative_residue_returned += staged_cumulative_residue
            self.annual_yield += staged_annual_yield
            self.harvest_count += len(events_today)
            accepted_time = torch.as_tensor(time).detach().to(dtype=torch.float64)
            self.last_harvest_time_seconds = self._scalar_from_value(
                accepted_time
            )

        raw_outputs = {
            "harvest_flux": harvest_flux,
            "harvest_biomass_carbon_input": carbon_input_flux,
            "harvest_carbon_export_flux": carbon_export_flux,
            "residue_c_flux": residue_c_flux,
            "harvest_biomass_nitrogen_input": nitrogen_input_flux,
            "harvest_nitrogen_export_flux": nitrogen_export_flux,
            "residue_n_flux": residue_n_flux,
            "delta_residue_cover": residue_cover_rate,
            "diagnostics.harvest.events_today": torch.as_tensor(
                len(events_today),
                device=zero.device,
                dtype=zero.dtype,
            ),
            "diagnostics.harvest.biomass_partition_residual": (
                biomass_partition_residual
            ),
            "diagnostics.harvest.carbon_ledger_residual": carbon_ledger_residual,
            "diagnostics.harvest.nitrogen_ledger_residual": (
                nitrogen_ledger_residual
            ),
            "diagnostics.harvest_management.free_energy": audit_outputs["free_energy"],
            "diagnostics.harvest_management.chemical_potential_norm": audit_outputs[
                "chemical_potential_norm"
            ],
        }
        raw_outputs = self._attach_canonical_runtime_diagnostic(raw_outputs)
        units = {
            "harvest_flux": "kg DM m-2 s-1",
            "harvest_biomass_carbon_input": "kg C m-2 s-1",
            "harvest_carbon_export_flux": "kg C m-2 s-1",
            "residue_c_flux": "kg C m-2 s-1",
            "harvest_biomass_nitrogen_input": "kg N m-2 s-1",
            "harvest_nitrogen_export_flux": "kg N m-2 s-1",
            "residue_n_flux": "kg N m-2 s-1",
            "delta_residue_cover": "s-1",
            "diagnostics.harvest.events_today": "1",
            "diagnostics.harvest.biomass_partition_residual": "kg DM m-2",
            "diagnostics.harvest.carbon_ledger_residual": "kg C m-2 s-1",
            "diagnostics.harvest.nitrogen_ledger_residual": "kg N m-2 s-1",
            "diagnostics.harvest_management.free_energy": "kg2 m-4",
            "diagnostics.harvest_management.chemical_potential_norm": "kg m-2",
            "diagnostics.harvest_management.canonical_iphs_runtime": "1",
        }
        carbon_fields = {
            "harvest_biomass_carbon_input",
            "harvest_carbon_export_flux",
            "residue_c_flux",
        }
        nitrogen_fields = {
            "harvest_biomass_nitrogen_input",
            "harvest_nitrogen_export_flux",
            "residue_n_flux",
        }
        return {
            name: UpdateItem(
                value=value,
                mode=(
                    UpdateMode.ACCUMULATE
                    if name == "delta_residue_cover"
                    else UpdateMode.REPLACE
                ),
                per_timestep=False,
                unit=units[name],
                element=(
                    "carbon"
                    if name in carbon_fields
                    else "nitrogen"
                    if name in nitrogen_fields
                    else None
                ),
                source_functor=self.name,
            )
            for name, value in raw_outputs.items()
        }

    def check_conservation_laws(
        self,
        input_state: Dict[str, torch.Tensor],
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return executable harvest partition and elemental-ledger residuals.

        References:
            Johnson et al. (2006), DOI:10.2134/agronj2005.0179.
        Mathematical expressions:
            r_B = |R_B| / max(|B_ag|, |R_B|)
            r_C = |R_C| / max(|F_C,in|, |F_C,out| + |F_C,res|)
            r_N = |R_N| / max(|F_N,in|, |F_N,out| + |F_N,res|)
        Parameter meanings:
            R_B is dry-matter partition residual [kg DM m-2]; R_C and R_N
            are elemental ledger residuals [kg element m-2 s-1].
        """
        # References: Johnson et al. (2006); RELIC local conservation hook contract.
        # Mathematical expression: mass_error = max(r_B, r_C, r_N).
        # Parameter meanings: R_B is biomass closure and R_C/R_N are elemental ledger residuals.
        # Units: r_B, r_C, r_N, and mass_error [1].
        # Domain/constraints: diagnostics and activity terms exist and are finite and non-empty.
        # Sign convention: returned mass_error is a non-negative magnitude; route signs remain in residuals.

        def _require_tensor(
            values: Mapping[str, Any],
            key: str,
        ) -> torch.Tensor:
            if key not in values:
                raise KeyError(f"Missing conservation value '{key}'")
            value = values[key]
            if isinstance(value, UpdateItem):
                value = value.value
            tensor = torch.as_tensor(value)
            if tensor.numel() == 0 or not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(
                    f"Harvest conservation value '{key}' must be finite and non-empty."
                )
            return tensor

        def _relative_residual(
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            absolute = torch.abs(residual)
            return torch.where(scale > 0.0, absolute / scale, absolute)

        biomass_residual = _require_tensor(
            output_updates,
            "diagnostics.harvest.biomass_partition_residual",
        )
        biomass_stock = _require_tensor(input_state, "aboveground_biomass").to(
            dtype=biomass_residual.dtype,
            device=biomass_residual.device,
        )
        biomass_scale = torch.maximum(
            torch.abs(biomass_stock),
            torch.abs(biomass_residual),
        )

        carbon_residual = _require_tensor(
            output_updates,
            "diagnostics.harvest.carbon_ledger_residual",
        )
        carbon_input = _require_tensor(
            output_updates,
            "harvest_biomass_carbon_input",
        )
        carbon_export = _require_tensor(
            output_updates,
            "harvest_carbon_export_flux",
        )
        carbon_residue = _require_tensor(output_updates, "residue_c_flux")
        carbon_scale = torch.maximum(
            torch.abs(carbon_input),
            torch.abs(carbon_export) + torch.abs(carbon_residue),
        )

        nitrogen_residual = _require_tensor(
            output_updates,
            "diagnostics.harvest.nitrogen_ledger_residual",
        )
        nitrogen_input = _require_tensor(
            output_updates,
            "harvest_biomass_nitrogen_input",
        )
        nitrogen_export = _require_tensor(
            output_updates,
            "harvest_nitrogen_export_flux",
        )
        nitrogen_residue = _require_tensor(output_updates, "residue_n_flux")
        nitrogen_scale = torch.maximum(
            torch.abs(nitrogen_input),
            torch.abs(nitrogen_export) + torch.abs(nitrogen_residue),
        )

        relative_residuals = (
            _relative_residual(biomass_residual, biomass_scale),
            _relative_residual(carbon_residual, carbon_scale),
            _relative_residual(nitrogen_residual, nitrogen_scale),
        )
        magnitude = torch.max(
            torch.stack(
                tuple(torch.max(value) for value in relative_residuals)
            )
        )
        return {"mass": self._scalar_from_value(magnitude)}

    def _check_conservation(
        self,
        *,
        total_aboveground: torch.Tensor,
        removed_biomass: torch.Tensor,
        residue_biomass: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check biomass partition conservation during harvest.

        References:
            - Lal (2005) Environment International 31, 575-584
        Mathematical expression:
            epsilon = max(0, |B_ag - (B_removed + B_residue)| - tol)
        Parameter meanings:
            B_ag: available aboveground biomass [kg m-2]
            B_removed: harvested/removed biomass [kg m-2]
            B_residue: retained residue biomass [kg m-2]
            tol: conservation tolerance [kg m-2]
        """
        # References: Lal (2005); Johnson et al. (2006).
        # Mathematical expression: R_B = max(0, |B_ag-B_removed-B_residue| - tol).
        # Parameter meanings: B_ag is available stock; removed/residue are partition branches.
        # Units: all biomass terms, tol, and R_B [kg DM m-2].
        # Domain/constraints: inputs are finite/non-negative and broadcast-compatible; tol >= 0.
        # Sign convention: R_B > 0 denotes partition mismatch; product export is outward-positive.
        tolerance = self._ensure_tensor(
            self.get_parameter_value(
                "conservation_tolerance",
                namespace="validation.conservation",
            ),
            dtype=total_aboveground.dtype,
        ).to(total_aboveground.device)
        closure_error = torch.abs(total_aboveground - (removed_biomass + residue_biomass))
        return torch.relu(closure_error - tolerance)

    def reset_annual_counters(self) -> None:
        """Reset annual harvest counters at start of year"""
        self.annual_yield = 0.0
        self.harvest_count = 0
    
    def get_harvest_statistics(self) -> Dict[str, float]:
        """
        Get harvest operation statistics
        
        Returns:
            Dictionary with harvest metrics
        """
        avg_yield = self.annual_yield / max(self.harvest_count, 1)
        
        return {
            "annual_yield_kg_ha": self.annual_yield * 10000,  # Convert to kg/ha
            "harvest_count": self.harvest_count,
            "average_yield_kg_ha": avg_yield * 10000,
            "cumulative_biomass_removed_kg_ha": self.cumulative_biomass_removed * 10000,
            "cumulative_c_removed_kg_ha": self.cumulative_c_removed * 10000,
            "cumulative_residue_returned_kg_ha": self.cumulative_residue_returned * 10000,
            "events_scheduled": len(self.harvest_events)
        }

    def _storage_potential(
        self,
        aboveground_biomass: torch.Tensor,
        belowground_biomass: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Harvest storage potential and chemical-potential effort of the biomass stocks.

        References:
            - Grmela and Ottinger (1997), GENERIC thermodynamic structure.
            - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical Expressions:
            x_h = [B_ag, B_ag, B_ag, B_bg, B_ag]^T
            G_h(x) = 0.5 x_h^T H_h x_h,  mu_h = H_h x_h
        """
        # References: Grmela and Ottinger (1997); van der Schaft and Maschke (1995).
        # Mathematical expression: x_h = [B_ag, B_ag, B_ag, B_bg, B_ag]; G_h = 0.5 x_h^T H_h x_h;
        #   mu_h = H_h x_h with H_h = diag(h) from build_hodge_contract.
        # Parameter meanings: B_ag, B_bg the aboveground and belowground biomass stocks; h the
        #   normalized harvest partition weights; G_h the storage potential; mu_h the effort.
        # Units: B, mu_h [kg m-2]; h [1]; G_h [kg2 m-4].
        # Domain/constraints: stocks are finite and non-negative (checked by the callers);
        #   h >= 0 with a positive sum; one weight per channel.
        # Sign convention: G_h >= 0; mu_h has the sign of the stocks (non-negative).
        hodge = self.build_hodge_contract(n_dofs=5, strict=True).matrix.to(
            device=aboveground_biomass.device,
            dtype=aboveground_biomass.dtype,
        )
        weights = torch.diag(hodge)
        cochain_state = torch.stack(
            (
                aboveground_biomass,
                aboveground_biomass,
                aboveground_biomass,
                belowground_biomass,
                aboveground_biomass,
            ),
            dim=0,
        )
        chemical_potential = weights.reshape((-1,) + (1,) * aboveground_biomass.ndim) * cochain_state
        free_energy = 0.5 * torch.sum(cochain_state * chemical_potential, dim=0)
        chemical_potential_norm = torch.linalg.vector_norm(
            chemical_potential.reshape(chemical_potential.shape[0], -1),
            dim=0,
        ).reshape_as(aboveground_biomass)
        return {
            "cochain_state": cochain_state,
            "chemical_potential": chemical_potential,
            "free_energy": free_energy,
            "chemical_potential_norm": chemical_potential_norm,
        }

    def _canonical_iphs_harvest_partition(
        self,
        state: Dict[str, torch.Tensor],
        *,
        event: HarvestEvent,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the canonical IPHS harvest partition of one scheduled event.

        References:
            - Lal (2005), crop residues and bioenergy use.
            - Johnson et al. (2006), residue management and GHG mitigation.
            - Grmela and Ottinger (1997), GENERIC thermodynamic structure.
            - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical Expressions:
            G_h(x)=0.5*x_h.T*H_h*x_h
            mu_h=dG_h/dx_h=H_h*x_h
            x_dot=M_h(u_h,theta)*mu_h, M_h=diag(u_h/h)
        Parameter meanings:
            H_h: ParameterManager-backed harvest Hodge weights h.
            x_h: biomass storage cochain [kg m-2].
            mu_h: harvest chemical potential or management effort.
            u_h: harvest controls of the scheduled event.
            theta: harvest efficiency and root mortality fraction.
        """
        # References: Lal (2005); Johnson et al. (2006); Grmela and Ottinger (1997).
        # Mathematical expression: u_H = clamp(HI_e eta_h q_e, 0, 1);
        #   u_R = clamp((1 - HI_e eta_h q_e) r_e, 0, 1); u_T = clamp(u_H + u_R, 0, 1);
        #   u_D = clamp(k_root, 0, 1); u = [u_H, u_R, u_T, u_D, u_R];
        #   x_dot = M_h mu_h = diag(u/h) H_h x_h = u * x_h; delta_cover = u_R where B_ag > 0.
        # Parameter meanings: HI_e, q_e, r_e the harvest index (clamped to [0, 1]), quality
        #   factor and residue retention of the scheduled event; eta_h the harvest
        #   efficiency; k_root the root mortality fraction; x_h the stock cochain; h the
        #   Hodge weights; delta_cover the residue-cover increment of the event.
        # Units: x_h, x_dot [kg m-2] per event (forward divides by one day); controls,
        #   HI_e, q_e, r_e, eta_h, k_root, delta_cover [1].
        # Domain/constraints: stocks finite and non-negative; the event is required (no
        #   default event, no event-less evaluation); the product u * x_h equals M_h mu_h
        #   for every positive weight and needs no floor for a zero weight.
        # Sign convention: grain export is outward-positive; residue/root return and the
        #   cover increment are return-positive.
        if not isinstance(event, HarvestEvent):
            raise TypeError(
                "The harvest partition evaluates one scheduled HarvestEvent 'event'; "
                f"got {type(event).__name__}"
            )

        target_dtype = torch.get_default_dtype()
        device = next(
            (v.device for v in state.values() if isinstance(v, torch.Tensor)),
            torch.device("cpu"),
        )

        def _required(name: str, units: str) -> torch.Tensor:
            # Declared inputs without a ParameterManager default: the biomass stocks
            # [kg m-2] and the residue cover [1]; a missing one raises.
            if state.get(name) is None:
                raise ValueError(
                    f"{self.name}: the harvest partition needs '{name}' [{units}] "
                    "as state or forcing; no default is applied."
                )
            return torch.as_tensor(state[name]).to(device=device, dtype=target_dtype)

        aboveground_biomass = _required("aboveground_biomass", "kg m-2")
        belowground_biomass = _required("belowground_biomass", "kg m-2")
        aboveground_biomass, belowground_biomass = torch.broadcast_tensors(
            aboveground_biomass,
            belowground_biomass,
        )
        residue_cover = torch.broadcast_to(
            _required("residue_cover", "1"),
            aboveground_biomass.shape,
        )
        crop_chars = self._get_crop_characteristics(event.crop_type)

        harvest_efficiency = torch.as_tensor(
            self.get_parameter_value(
                "harvest_efficiency",
                namespace="anthroposphere.management.harvest",
                required=True,
            ),
            device=device,
            dtype=target_dtype,
        )
        root_mortality_rate = torch.as_tensor(
            self.get_parameter_value(
                "root_mortality_rate",
                namespace="anthroposphere.management.harvest",
                required=True,
            ),
            device=device,
            dtype=target_dtype,
        )
        effective_hi = torch.clamp(
            torch.as_tensor(event.harvest_index, device=device, dtype=target_dtype),
            min=0.0,
            max=1.0,
        )
        quality_factor = torch.as_tensor(
            event.quality_factor,
            device=device,
            dtype=target_dtype,
        )
        residue_retention = torch.as_tensor(
            event.residue_retention,
            device=device,
            dtype=target_dtype,
        )
        removed_fraction = effective_hi * harvest_efficiency * quality_factor
        harvest_control = torch.ones_like(aboveground_biomass) * torch.clamp(
            removed_fraction,
            min=0.0,
            max=1.0,
        )
        residue_control = torch.ones_like(aboveground_biomass) * torch.clamp(
            (1.0 - removed_fraction) * residue_retention,
            min=0.0,
            max=1.0,
        )
        total_control = torch.clamp(
            harvest_control + residue_control,
            min=0.0,
            max=1.0,
        )
        root_control = torch.ones_like(aboveground_biomass) * torch.clamp(
            root_mortality_rate,
            min=0.0,
            max=1.0,
        )

        potential = self._storage_potential(aboveground_biomass, belowground_biomass)
        control_vector = torch.stack(
            (
                harvest_control,
                residue_control,
                total_control,
                root_control,
                residue_control,
            ),
            dim=0,
        )
        state_advancer = control_vector * potential["cochain_state"]

        grain_yield_rate = state_advancer[0]
        residue_generation_rate = state_advancer[1]
        biomass_removal_rate = state_advancer[2]
        root_mortality_flux = state_advancer[3]
        residue_cover_change = torch.where(
            aboveground_biomass > 0.0,
            residue_control,
            torch.zeros_like(residue_control),
        )
        residue_cover_after = torch.clamp(residue_cover + residue_cover_change, max=1.0)
        c_content = torch.as_tensor(
            crop_chars["c_content"],
            device=device,
            dtype=target_dtype,
        )
        belowground_carbon_flux = root_mortality_flux * c_content

        return {
            "biomass_removal_rate": biomass_removal_rate,
            "grain_yield_rate": grain_yield_rate,
            "effective_harvest_index": effective_hi,
            "residue_generation_rate": residue_generation_rate,
            "delta_residue_cover": residue_cover_change,
            "residue_cover_after_harvest": residue_cover_after,
            "root_mortality_flux": root_mortality_flux,
            "belowground_carbon_flux": belowground_carbon_flux,
            "state_advancer": state_advancer,
            "chemical_potential": potential["chemical_potential"],
            "free_energy": potential["free_energy"],
            "chemical_potential_norm": potential["chemical_potential_norm"],
        }


# Export all public classes and functions
__all__ = [
    'HarvestManagement',
    'CropType',
    'HarvestEvent',
]



