"""
Land Use Transitions Module - Anthroposphere Component

This module implements land-use dynamics with a continuous-time Markov chain
and optional discrete events. It uses LUH2-class categories and calibrated
transition channels; the socioeconomic driver scaling is phenomenological.

References:
    - Hurtt, G.C., et al. (2020). Harmonization of global land use change and management for the period 850-2100 (LUH2) for CMIP6. GMD 13, 5425-5464.
    - IPCC (2019). Climate Change and Land: IPCC Special Report.
    - Verburg, P.H., et al. (2009). Combining top-down and bottom-up dynamics in land use modeling. Landscape Ecology 24(9), 1167-1181.

Mathematical Expression:
    Land use transitions follow continuous-time Markov chain dynamics:
    dP/dt = P * Q
    P(t+dt) = P(t) * exp(Q * dt_years)
    
    where:
    - P(t) is the state probability vector at time t [dimensionless]
    - Q is the generator matrix at time t [yr^-1]
    - Generator constraint: sum_j Q[i,j] = 0 for all i
    
    Conservation constraint:
    sum_i P[i] = 1 (total land-area fraction must equal 1)
"""

import torch
from typing import Any, Dict, List, Mapping, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from relic.core.functor import ScaleCouplingFunctor
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import Tensor
from relic.core.state import UpdateItem, UpdateMode
from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.utils.physical_constants import get_physical_constant

# Parameter management imports

logger = logging.getLogger(__name__)


class LandUseType(Enum):
    """
    Land use types following LUH2/CMIP6 classification
    
    References:
        - Hurtt et al. (2020). LUH2 for CMIP6
        - IPCC (2019). Land use classification system
    """
    PRIMARY_FOREST = "primary_forest"
    SECONDARY_FOREST = "secondary_forest"
    CROPLAND_C3 = "cropland_c3"  # C3 crops (wheat, rice)
    CROPLAND_C4 = "cropland_c4"  # C4 crops (maize, sugarcane)
    PASTURE = "pasture"
    RANGELAND = "rangeland"
    URBAN = "urban"
    WATER = "water"
    BARREN = "barren"


class TransitionType(Enum):
    """Types of land use transitions"""
    DEFORESTATION = "deforestation"
    AFFORESTATION = "afforestation"
    URBANIZATION = "urbanization"
    AGRICULTURAL_EXPANSION = "agricultural_expansion"
    AGRICULTURAL_ABANDONMENT = "agricultural_abandonment"
    CROP_ROTATION = "crop_rotation"
    RESTORATION = "restoration"


@dataclass
class TransitionEvent:
    """
    Data class for land use transition events
    
    Attributes:
        from_type: Source land use type
        to_type: Target land use type
        area_fraction: Fraction of area to transition [1]
        time_index: Time index for event-based transitions
        spatial_mask: Optional spatial mask for heterogeneous transitions
    """
    from_type: LandUseType
    to_type: LandUseType
    area_fraction: float  # [1]
    time_index: Optional[int] = None
    spatial_mask: Optional[torch.Tensor] = None


class LandUseTransitions(ScaleCouplingFunctor):
    """
    Land Use Transitions Functor
    
    Advances the canonical nine-class ``land_use_fraction`` state with one
    continuous-time Markov-chain generator and optional discrete events.
    Socioeconomic controls scale the calibrated generator, while its row-sum
    zero constraint preserves total land-area fraction on every support.
    
    Parameters:
        transition_rates: Base land-use transition rates [yr-1]
        transition_channels: Source-to-target split fractions [1]
        numerical: Calibratable socioeconomic driver coefficients [1]
        soil: Class-specific soil-carbon input rates [kg C m-2 yr-1]
        transition_events: Optional externally registered transition events
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    INPUT_VARIABLES = [
        "land_use_fraction",
        "time_index",
        "gdp_per_capita",
        "population_density",
        "market_accessibility",
        "policy_intensity",
    ]
    OUTPUT_UNITS = {
        "land_use_fraction": "1",
        "diagnostics.land_use_transitions.delta_fraction": "1",
        "diagnostics.land_use_transitions.dominant_type_index": "1",
        "diagnostics.land_use_transitions.maximum_fraction": "1",
        "diagnostics.land_use_transitions.soil_carbon_input_rate_change": (
            "kg C m-2 s-1"
        ),
        "diagnostics.land_use_transitions.canonical_iphs_runtime": "1",
    }
    DRIVER_INPUTS = (
        "gdp_per_capita",
        "population_density",
        "market_accessibility",
        "policy_intensity",
    )
    
    def __init__(
        self,
        name: Union[str, Dict[str, Any]] = "land_use_transitions",
        config: Optional[Dict[str, Any]] = None,
        *,
        upscale_bridge: Optional[Any] = None,
        downscale_bridge: Optional[Any] = None,
    ):
        """
        Initialize land use transitions functor
        
        Args:
            name: Functor name
            config: Configuration dictionary
            upscale_bridge: Bound fine-to-coarse typed chain map.
            downscale_bridge: Bound coarse-to-fine typed chain map.
        """
        if upscale_bridge is not None and not callable(upscale_bridge):
            raise TypeError("upscale_bridge must be callable")
        if downscale_bridge is not None and not callable(downscale_bridge):
            raise TypeError("downscale_bridge must be callable")
        if isinstance(name, dict):
            incoming_config = name
            name = incoming_config.get("transitions_name", "land_use_transitions")
            if config is not None:
                merged_config = {**incoming_config, **config}
            else:
                merged_config = incoming_config
            config = merged_config
        # Set up metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "LUH2-class CTMC land-use transitions with calibratable "
                "phenomenological driver scaling"
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.PREPROCESSING,
            mathematical_expression=(
                "p_dot = p Q(theta); p(t+dt) = p exp(Q dt_years); "
                "Q 1 = 0; H_Q = diag(sum_{j != i} Q_ij) >= 0"
            ),
            references=[
                "Hurtt et al. (2020) DOI:10.5194/gmd-13-5425-2020",
                "IPCC (2019) Climate Change and Land Report",
                "Verburg et al. (2009) DOI:10.1007/s10980-009-9347-7"
            ],
            parameters={
                "anthroposphere.land_use.transition_rates.*": (
                    "Calibrated CTMC transition rates [yr-1]"
                ),
                "anthroposphere.land_use.transition_channels.*": (
                    "Source-to-target split fractions [1]"
                ),
                "anthroposphere.land_use.numerical.*": (
                    "Driver scaling and numerical metric parameters [1]"
                ),
                "anthroposphere.land_use.soil.*": (
                    "Class-specific soil-carbon input rates [kg C m-2 yr-1]"
                ),
                "validation.conservation.conservation_tolerance": (
                    "Land-area fraction closure tolerance [1]"
                ),
            },
            input_variables=list(self.INPUT_VARIABLES),
            output_variables=list(self.OUTPUT_UNITS),
            conservation_laws=["land_area_fraction"],
            spatial_dimensions=2,
            temporal_scale=TemporalScale.YEAR,
            coupling_interfaces=["mapping", "biosphere", "pedosphere"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        # Default configuration; numeric land-use transition parameters are managed in the parameter database
        default_config: Dict[str, Any] = {
            "dependencies": []
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        # Scale coupling: land-use fraction cochains move between the declared fine
        # and coarse grid supports through the bound conservative chain maps
        # (upscale R_h, downscale P_h).
        super().__init__(
            name,
            default_config,
            metadata,
            source_scale="fine_grid",
            target_scale="coarse_grid",
            coupling_type="spatial",
        )
        self.runtime_path = "canonical_iphs_land_use_ctmc_chain_map"
        self._upscale_bridge = upscale_bridge
        self._downscale_bridge = downscale_bridge
        
        # Initialize land use types
        self.land_use_types = list(LandUseType)
        self.n_types = len(self.land_use_types)
        
        # Initialize transition matrix
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Store transition events
        self.transition_events: List[TransitionEvent] = []
        
    
    def _initialize_transition_matrix(self) -> torch.Tensor:
        """
        Initialize transition matrix with default rates
        
        Returns:
            Transition matrix [n_types * n_types] with rates [yr-1]
        """
        matrix = torch.zeros((self.n_types, self.n_types))
        # Constraint (Hurtt et al. 2020 LUH2): channel fractions must satisfy sum(f_i)=1 to conserve total deforestation/afforestation flows.

        # Get indices for each land use type
        primary_idx = self.land_use_types.index(LandUseType.PRIMARY_FOREST)
        secondary_idx = self.land_use_types.index(LandUseType.SECONDARY_FOREST)
        crop_c3_idx = self.land_use_types.index(LandUseType.CROPLAND_C3)
        crop_c4_idx = self.land_use_types.index(LandUseType.CROPLAND_C4)
        pasture_idx = self.land_use_types.index(LandUseType.PASTURE)
        urban_idx = self.land_use_types.index(LandUseType.URBAN)
        
        # Deforestation transitions
        # Reference: Hurtt et al. (2011) LUH2
        deforest_rate = self.get_parameter_value(
            "transition_rates.deforestation_rate",
            namespace="anthroposphere.land_use",
        )
        f_pri_sec = self.get_parameter_value(
            "transition_channels.primary_to_secondary",
            namespace="anthroposphere.land_use",
        )
        f_pri_crop = self.get_parameter_value(
            "transition_channels.primary_to_crops",
            namespace="anthroposphere.land_use",
        )
        f_pri_pas = self.get_parameter_value(
            "transition_channels.primary_to_pasture",
            namespace="anthroposphere.land_use",
        )
        f_pri_sec, f_pri_crop, f_pri_pas = self._validate_channel_fractions(
            [f_pri_sec, f_pri_crop, f_pri_pas],
            "primary_deforestation_channels",
        )

        matrix[primary_idx, secondary_idx] = deforest_rate * f_pri_sec
        matrix[primary_idx, crop_c3_idx] = deforest_rate * f_pri_crop
        matrix[primary_idx, pasture_idx] = deforest_rate * f_pri_pas

        f_sec_crop = self.get_parameter_value(
            "transition_channels.secondary_to_crops",
            namespace="anthroposphere.land_use",
        )
        f_sec_pas = self.get_parameter_value(
            "transition_channels.secondary_to_pasture",
            namespace="anthroposphere.land_use",
        )

        f_sec_crop, f_sec_pas = self._validate_channel_fractions(
            [f_sec_crop, f_sec_pas],
            "secondary_deforestation_channels",
        )

        matrix[secondary_idx, crop_c3_idx] = deforest_rate * f_sec_crop
        matrix[secondary_idx, pasture_idx] = deforest_rate * f_sec_pas

        # Afforestation transitions
        afforest_rate = self.get_parameter_value(
            "transition_rates.afforestation_rate",
            namespace="anthroposphere.land_use",
        )
        
        f_crop_sec = self.get_parameter_value(
            "transition_channels.crops_to_secondary",
            namespace="anthroposphere.land_use",
        )
        f_pas_sec = self.get_parameter_value(
            "transition_channels.pasture_to_secondary",
            namespace="anthroposphere.land_use",
        )
        f_crop_sec, f_pas_sec = self._validate_channel_fractions(
            [f_crop_sec, f_pas_sec],
            "afforestation_channels",
        )
        matrix[crop_c3_idx, secondary_idx] = afforest_rate * f_crop_sec
        matrix[pasture_idx, secondary_idx] = afforest_rate * f_pas_sec

        # Urbanization (from any non-urban)
        urban_rate = self.get_parameter_value(
            "transition_rates.urbanization_rate",
            namespace="anthroposphere.land_use",
        )
        for i in range(self.n_types):
            if i != urban_idx:
                matrix[i, urban_idx] = urban_rate

        # Agricultural transitions
        ag_expand = self.get_parameter_value(
            "transition_rates.agricultural_expansion_rate",
            namespace="anthroposphere.land_use",
        )
        
        f_pas_c3 = self.get_parameter_value(
            "transition_channels.pasture_to_crops_c3",
            namespace="anthroposphere.land_use",
        )
        f_pas_c4 = self.get_parameter_value(
            "transition_channels.pasture_to_crops_c4",
            namespace="anthroposphere.land_use",
        )
        f_pas_c3, f_pas_c4 = self._validate_channel_fractions(
            [f_pas_c3, f_pas_c4],
            "agricultural_expansion_channels",
        )
        matrix[pasture_idx, crop_c3_idx] = ag_expand * f_pas_c3
        matrix[pasture_idx, crop_c4_idx] = ag_expand * f_pas_c4

        # Abandonment
        abandon_rate = self.get_parameter_value(
            "transition_rates.abandonment_rate",
            namespace="anthroposphere.land_use",
        )
        matrix[crop_c3_idx, pasture_idx] = abandon_rate
        matrix[crop_c4_idx, pasture_idx] = abandon_rate
        
        return self._build_ctmc_generator(matrix)

    def _build_ctmc_generator(self, offdiag_rates: torch.Tensor) -> torch.Tensor:
        """
        Build CTMC generator matrix Q from off-diagonal transition rates.

        References: Hurtt et al. (2020); Essentials of Stochastic Processes.
        Mathematical expression:
            Q_ij = r_ij >= 0, i != j
            Q_ii = -sum_{j!=i} Q_ij
        Parameters:
            offdiag_rates: candidate off-diagonal rates [yr^-1]
        """
        if not torch.isfinite(offdiag_rates).all():
            raise ValueError("CTMC transition rates must be finite")
        q = offdiag_rates.clone()
        q.fill_diagonal_(0.0)
        if torch.any(q < 0.0):
            raise ValueError("CTMC off-diagonal transition rates must be non-negative")
        row_sums = q.sum(dim=1)
        q = q - torch.diag(row_sums)
        return q

    def _validate_channel_fractions(
        self,
        fractions: List[Union[float, torch.Tensor]],
        channel_name: str,
        atol: float = 1e-6,
    ) -> List[torch.Tensor]:
        """
        Validate transition channel fractions against sum(f_i) = 1.

        References: Hurtt et al. (2020) LUH2; conservation constraint for land-use transition split fractions.
        Mathematical expression: sum_i f_i = 1 with f_i >= 0.
        Parameters: fractions (dimensionless split weights for channel_name), atol (tolerance for deviation).
        """
        tensor_fracs = [torch.as_tensor(f, dtype=torch.float32) for f in fractions]
        stacked = torch.stack(tensor_fracs)
        if torch.any(stacked < 0):
            raise ValueError(f"{channel_name} contains negative fractions.")
        total = stacked.sum()
        if torch.abs(total - 1.0) > atol:
            raise ValueError(
                f"{channel_name} fractions must sum to one; received {float(total):.6f}"
            )
        return [stacked[i] for i in range(stacked.shape[0])]
    
    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Resolve structural audit size without changing runtime tensor shape."""

        value = self.n_types if n_dofs is None else int(n_dofs)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare cochain spaces for the land-use CTMC chain map.

        References:
          - Hurtt et al. (2020), LUH2 land-use transition harmonization.
          - Verburg et al. (2009), land-use change transition models.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          p_dot = p Q(theta), p(t+dt) = p exp(Q dt), and Q 1 = 0.
        Parameter meanings:
          p is the land-use fraction cochain [1], Q is the ParameterManager
          generated CTMC rate matrix [yr^-1], and p_dot is the transition flow
          cochain [yr^-1].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_land_use_transitions",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="1",
                extensivity="integral",
                physical_domain="anthroposphere",
                semantic_name="land_use_fraction_state",
                orientation_convention="area_fraction_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_land_use_transitions",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="yr^-1",
                extensivity="density",
                physical_domain="anthroposphere",
                semantic_name="land_use_transition_flux",
                orientation_convention="from_source_to_target",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_land_use_transitions",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="yr^-1",
                extensivity="density",
                physical_domain="anthroposphere",
                semantic_name="land_use_transition_port_flow",
                orientation_convention="into_land_use_state",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager namespaces used by runtime and structure."""

        return (
            "anthroposphere.land_use.transition_rates.*",
            "anthroposphere.land_use.transition_channels.*",
            "anthroposphere.land_use.numerical.*",
            "anthroposphere.land_use.soil.*",
            "validation.conservation.conservation_tolerance",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive transition mobility Hodge contract.

        References:
          - Hurtt et al. (2020), LUH2 transition rates.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
          - Ramirez and Le Gorrec (2022), irreversible port-Hamiltonian systems.
        Mathematical expression:
          H_Q = diag(lambda_i), lambda_i = sum_{j != i} Q_ij >= 0.
        Parameter meanings:
          Q_ij is the land-use transition generator coefficient [yr^-1] and
          lambda_i is the non-negative exit mobility from land-use class i.
        """

        n_dofs = self._structural_n_dofs(n_dofs)
        matrix_q = self.transition_matrix.to(
            device=getattr(self, "device", torch.device("cpu")),
            dtype=torch.float32,
        )
        if matrix_q.shape[0] != n_dofs:
            if matrix_q.shape[0] > n_dofs:
                matrix_q = matrix_q.narrow(0, 0, n_dofs).narrow(1, 0, n_dofs)
            else:
                pad_rows = torch.zeros(
                    n_dofs - matrix_q.shape[0],
                    matrix_q.shape[1],
                    device=matrix_q.device,
                    dtype=matrix_q.dtype,
                )
                matrix_q = torch.cat([matrix_q, pad_rows], dim=0)
                pad_cols = torch.zeros(
                    n_dofs,
                    n_dofs - matrix_q.shape[1],
                    device=matrix_q.device,
                    dtype=matrix_q.dtype,
                )
                matrix_q = torch.cat([matrix_q, pad_cols], dim=1)
        off_diag = matrix_q.clone()
        off_diag.fill_diagonal_(0.0)
        if not torch.isfinite(off_diag).all():
            raise ValueError("CTMC Hodge rates must be finite")
        if torch.any(off_diag < 0.0):
            raise ValueError("CTMC Hodge off-diagonal rates must be non-negative")
        exit_rates = off_diag.sum(dim=1)
        hodge_matrix = torch.diag(exit_rates)
        domain = CochainSpaceSpec(
            complex_id="anthroposphere_land_use_transitions",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="1",
            extensivity="integral",
            physical_domain="anthroposphere",
            semantic_name="land_use_fraction_state",
            orientation_convention="area_fraction_positive",
            boundary_role="interior",
        )
        codomain = CochainSpaceSpec(
            complex_id="anthroposphere_land_use_transitions",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="yr^-1",
            extensivity="density",
            physical_domain="anthroposphere",
            semantic_name="land_use_transition_flux",
            orientation_convention="from_source_to_target",
            boundary_role="interior",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_ctmc_mobility_hodge",
            domain=domain,
            codomain=codomain,
            matrix=hodge_matrix,
            parameter_keys=(
                "anthroposphere.land_use.transition_rates.*",
                "anthroposphere.land_use.transition_channels.*",
            ),
            metric_keys=("anthroposphere.land_use.numerical.unity_factor",),
            material_law="land_use_ctmc_exit_mobility",
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
        """Return no independent energy storage for the land-transition map."""

        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the CTMC entropy support audit term."""

        _ = state
        return {
            "land_use_transition_entropy": "sigma = z.T H_Q z >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the conservative CTMC chain-map metadata."""

        _ = state, complex_bundle
        return {
            "state_vars": tuple(land_use_type.value for land_use_type in self.land_use_types),
            "chain_map": "p_dot = p Q(theta)",
            "row_sum_zero": True,
            "mass_conserving": True,
            "ports": self.ports(),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive CTMC mobility operator for audits."""

        _ = state, complex_bundle
        hodge = self.build_hodge_contract(strict=True)
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"land_use_fraction_state[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the land-use transition management port."""

        effort_space = CochainSpaceSpec(
            complex_id="anthroposphere_land_use_transitions",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="1",
            extensivity="intensive",
            physical_domain="anthroposphere",
            semantic_name="land_use_transition_potential",
            orientation_convention="into_land_use_state",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="anthroposphere_land_use_transitions",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="yr^-1",
            extensivity="density",
            physical_domain="anthroposphere",
            semantic_name="land_use_transition_port_flow",
            orientation_convention="into_land_use_state",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="land_use_transition_area_fraction_port",
                effort_var="land_use_transition_potential",
                flow_var="land_use_transition_port_flow",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="yr^-1",
                positive_power_convention="into_system",
                port_kind="management",
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

    def _land_use_type_axis(self, fractions: torch.Tensor) -> int:
        """Return the declared class axis for supported canonical tensor layouts."""
        if fractions.dim() == 1:
            return 0
        if fractions.dim() == 2:
            return 1
        if fractions.dim() == 3:
            candidates = [
                axis for axis in (0, 1) if fractions.shape[axis] == self.n_types
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Three-dimensional land_use_fraction must have one unique "
                    "nine-class axis in position 0 or 1"
                )
            return candidates[0]
        if fractions.dim() == 4:
            return 1
        raise ValueError(
            "land_use_fraction must use [type], [batch,type], [type,y,x], "
            "[batch,type,space], or [batch,type,y,x] layout"
        )

    def _validate_fraction_state(
        self,
        state: Mapping[str, Any],
    ) -> tuple[torch.Tensor, int]:
        """Validate the canonical land-use state and optional external drivers."""
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if "land_use_fraction" not in state:
            raise KeyError("Missing required input: land_use_fraction")
        fractions = state["land_use_fraction"]
        if not isinstance(fractions, torch.Tensor):
            raise TypeError("land_use_fraction must be a torch.Tensor")
        if not torch.is_floating_point(fractions):
            raise TypeError("land_use_fraction must use a floating-point dtype")
        if fractions.numel() == 0:
            raise ValueError("land_use_fraction must not be empty")
        if not torch.isfinite(fractions).all():
            raise ValueError("land_use_fraction must contain only finite values")
        if torch.any(fractions < 0.0):
            raise ValueError("land_use_fraction must be non-negative")

        type_axis = self._land_use_type_axis(fractions)
        if fractions.shape[type_axis] != self.n_types:
            raise ValueError(
                f"land_use_fraction must contain exactly {self.n_types} "
                "land-use classes"
            )

        tolerance = float(
            self.get_parameter_value(
                "conservation_tolerance",
                namespace="validation.conservation",
                source_order=("manager",),
                required=True,
            )
        )
        totals = fractions.sum(dim=type_axis)
        if torch.max(torch.abs(totals - 1.0)) > tolerance:
            raise ValueError("land_use_fraction must sum to one on every support")

        for name in self.DRIVER_INPUTS:
            if name not in state:
                continue
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.numel() == 0:
                raise ValueError(f"{name} must not be empty")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            if torch.any(value < 0.0):
                raise ValueError(f"{name} must be non-negative")
            if name in {"market_accessibility", "policy_intensity"} and torch.any(
                value > 1.0
            ):
                raise ValueError(f"{name} must lie within [0, 1]")

        if "time_index" in state:
            time_index = state["time_index"]
            if not isinstance(time_index, torch.Tensor):
                raise TypeError("time_index must be a torch.Tensor")
            if time_index.numel() != 1 or not torch.isfinite(time_index).all():
                raise ValueError("time_index must be one finite scalar")
            time_index_value = float(time_index.detach().reshape(-1)[0].item())
            if time_index_value < 0.0 or not time_index_value.is_integer():
                raise ValueError("time_index must be a non-negative integer scalar")

        return fractions, type_axis

    def validate_state(self, state: Dict[str, torch.Tensor]) -> List[str]:
        """Return canonical state validation issues for lifecycle audits."""
        try:
            self._validate_fraction_state(state)
        except (KeyError, TypeError, ValueError) as exc:
            return [str(exc)]
        return []
    
    def add_transition_event(self, event: TransitionEvent) -> None:
        """
        Add a discrete transition event
        
        Args:
            event: TransitionEvent with transition details
        """
        if not isinstance(event, TransitionEvent):
            raise TypeError("event must be a TransitionEvent")
        area_fraction = self._scalar_value(event.area_fraction, "event.area_fraction")
        self._validate_parameter_bounds("event.area_fraction", area_fraction)
        if event.time_index is not None and int(event.time_index) < 0:
            raise ValueError("event.time_index must be non-negative")
        if event.spatial_mask is not None:
            if not isinstance(event.spatial_mask, torch.Tensor):
                raise TypeError("event.spatial_mask must be a torch.Tensor")
            if not torch.isfinite(event.spatial_mask).all():
                raise ValueError("event.spatial_mask must contain only finite values")
            if torch.any((event.spatial_mask < 0.0) | (event.spatial_mask > 1.0)):
                raise ValueError("event.spatial_mask must lie within [0, 1]")
        self.transition_events.append(event)

    def _validate_parameter_bounds(self, name: str, value: float) -> None:
        """Validate parameter bounds for probability-like variables."""
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be within [0,1]")

    @staticmethod
    def _scalar_value(value: Any, name: str = "value") -> float:
        """Convert scalar-like values to finite floats."""
        if isinstance(value, torch.Tensor):
            detached = value.detach()
            if detached.numel() != 1:
                raise ValueError(f"{name} must contain exactly one scalar value.")
            scalar = float(detached.reshape(-1)[0].tolist())
        else:
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be scalar-convertible.") from exc
        if not torch.isfinite(torch.as_tensor(scalar)):
            raise ValueError(f"{name} must be finite.")
        return scalar

    def _compute_transition_matrix(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build the canonical CTMC generator Q [yr^-1]."""
        fractions = state["land_use_fraction"]
        base_matrix = self.transition_matrix.to(
            device=fractions.device,
            dtype=fractions.dtype,
        )
        # References: Hurtt et al. (2020) LUH2; Verburg et al. (2009)
        # Math: scale = base + w_gdp * log1p(mean(gdp)/gdp_ref) + w_pop * log1p(mean(pop)/pop_ref)
        #       + w_market * mean(market); scale = scale * max(1 - mean(policy), policy_min)
        # Parameters: driver_scale_base, driver_gdp_weight, driver_gdp_reference,
        #             driver_population_weight, driver_population_reference, driver_market_weight, driver_policy_min
        driver_scale_base = self.get_parameter_value(
            "numerical.driver_scale_base",
            namespace="anthroposphere.land_use",
            source_order=("manager",), required=True)
        driver_gdp_weight = self.get_parameter_value(
            "numerical.driver_gdp_weight",
            namespace="anthroposphere.land_use",
            source_order=("manager",), required=True)
        driver_gdp_reference = self.get_parameter_value(
            "numerical.driver_gdp_reference",
            namespace="anthroposphere.land_use",
            source_order=("manager",), required=True)
        driver_population_weight = self.get_parameter_value(
            "numerical.driver_population_weight",
            namespace="anthroposphere.land_use",
            source_order=("manager",), required=True)
        driver_population_reference = self.get_parameter_value(
            "numerical.driver_population_reference",
            namespace="anthroposphere.land_use",
            source_order=("manager",), required=True)
        driver_market_weight = self.get_parameter_value(
            "numerical.driver_market_weight",
            namespace="anthroposphere.land_use",
            source_order=("manager",), required=True)
        driver_policy_min = self.get_parameter_value(
            "numerical.driver_policy_min",
            namespace="anthroposphere.land_use",
            source_order=("manager",), required=True)
        driver_scale = torch.as_tensor(
            driver_scale_base,
            device=fractions.device,
            dtype=fractions.dtype,
        )
        driver_market_weight_tensor = torch.as_tensor(
            driver_market_weight,
            device=fractions.device,
            dtype=fractions.dtype,
        )
        if "gdp_per_capita" in state:
            gdp = state["gdp_per_capita"].to(
                device=fractions.device,
                dtype=fractions.dtype,
            )
            driver_scale = driver_scale + float(driver_gdp_weight) * torch.log1p(
                torch.mean(gdp) / float(driver_gdp_reference)
            )
        if "population_density" in state:
            population = state["population_density"].to(
                device=fractions.device,
                dtype=fractions.dtype,
            )
            driver_scale = driver_scale + float(driver_population_weight) * torch.log1p(
                torch.mean(population) / float(driver_population_reference)
            )
        if "market_accessibility" in state:
            market = state["market_accessibility"].to(
                device=fractions.device,
                dtype=fractions.dtype,
            )
            driver_scale = driver_scale + driver_market_weight_tensor * torch.mean(
                market
            )
        if "policy_intensity" in state:
            policy = state["policy_intensity"].to(
                device=fractions.device,
                dtype=fractions.dtype,
            )
            driver_scale = driver_scale * torch.clamp(
                1.0 - torch.mean(policy),
                min=float(driver_policy_min),
            )
        diag = torch.diag(torch.diagonal(base_matrix))
        off_diag = base_matrix - diag
        off_diag = off_diag * driver_scale
        return self._build_ctmc_generator(off_diag)

    def _get_soil_carbon_inputs(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Fetch soil carbon input rates per land use type [kg C m-2 yr-1]."""
        if not hasattr(self, "_soil_carbon_inputs_cache"):
            inputs: List[float] = []
            for lu_type in self.land_use_types:
                param_key = f"soil.{lu_type.value}.soil_carbon_input"
                inputs.append(
                    float(
                        self.get_parameter_value(
                            param_key,
                            namespace="anthroposphere.land_use",
                            source_order=("manager",), required=True)
                    )
                )
            self._soil_carbon_inputs_cache = torch.tensor(inputs, dtype=torch.float32)
        return self._soil_carbon_inputs_cache.to(device=device, dtype=dtype)

    @staticmethod
    def _unwrap_update_value(value: Any) -> Any:
        """Extract tensor payload from UpdateItem-like wrappers."""
        if hasattr(value, "value"):
            return value.value
        return value

    def _build_runtime_outputs(
        self,
        previous_fractions: torch.Tensor,
        updated_fractions: torch.Tensor,
        type_axis: int,
    ) -> Dict[str, UpdateItem]:
        """Build the exact canonical full-state and observation contract."""
        delta_fraction = updated_fractions - previous_fractions
        soil_carbon_inputs = self._get_soil_carbon_inputs(
            updated_fractions.device,
            updated_fractions.dtype,
        )
        reshape_dims = [1] * updated_fractions.dim()
        reshape_dims[type_axis] = self.n_types
        soil_carbon_inputs = soil_carbon_inputs.view(*reshape_dims)
        seconds_per_year = float(get_physical_constant("seconds_per_year"))

        # References: Hurtt et al. (2020), LUH2 land-use harmonization.
        # Mathematical expression: Delta I_C = sum_i(Delta p_i I_C,i) / tau_yr.
        # Parameter meanings: Delta p_i is the per-step class-fraction change [1],
        # I_C,i is the class soil-carbon input rate [kg C m-2 yr-1], and
        # tau_yr is the physical-constant conversion [s yr-1].
        soil_carbon_input_rate_change = (
            delta_fraction * soil_carbon_inputs
        ).sum(dim=type_axis) / seconds_per_year
        values = {
            "land_use_fraction": updated_fractions,
            "diagnostics.land_use_transitions.delta_fraction": delta_fraction,
            "diagnostics.land_use_transitions.dominant_type_index": torch.argmax(
                updated_fractions,
                dim=type_axis,
            ),
            "diagnostics.land_use_transitions.maximum_fraction": torch.max(
                updated_fractions,
                dim=type_axis,
            ).values,
            "diagnostics.land_use_transitions.soil_carbon_input_rate_change": (
                soil_carbon_input_rate_change
            ),
            "diagnostics.land_use_transitions.canonical_iphs_runtime": (
                updated_fractions.new_ones(())
            ),
        }
        if set(values) != set(self.OUTPUT_UNITS):
            missing = sorted(set(self.OUTPUT_UNITS) - set(values))
            extra = sorted(set(values) - set(self.OUTPUT_UNITS))
            raise RuntimeError(
                f"LandUseTransitions output contract mismatch: missing={missing}, "
                f"extra={extra}"
            )
        return {
            name: UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit=self.OUTPUT_UNITS[name],
                source_functor=self.name,
            )
            for name, value in values.items()
        }

    def _check_conservation_laws(self, state: Dict[str, torch.Tensor], updates: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Verify area conservation (sum of fractions = 1)."""
        errors: Dict[str, float] = {}
        fractions = self._unwrap_update_value(updates.get("land_use_fraction"))
        delta = self._unwrap_update_value(
            updates.get("diagnostics.land_use_transitions.delta_fraction")
        )
        if fractions is None or delta is None:
            raise KeyError("Canonical land-use conservation outputs are missing")
        fractions_tensor = torch.as_tensor(fractions, dtype=torch.float64)
        delta_tensor = torch.as_tensor(delta, dtype=torch.float64)
        type_axis = self._land_use_type_axis(fractions_tensor)
        fraction_residual = torch.max(
            torch.abs(fractions_tensor.sum(dim=type_axis) - 1.0)
        )
        delta_residual = torch.max(torch.abs(delta_tensor.sum(dim=type_axis)))
        errors["area_fraction_conservation"] = float(fraction_residual)
        errors["area_increment_conservation"] = float(delta_residual)
        return errors

    def check_conservation_laws(
        self,
        input_state: Dict[str, torch.Tensor],
        output_updates: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Expose the land-area ledger to the unified Functor lifecycle."""
        # References:
        #   - Hurtt et al. (2020), LUH2 land-use transition harmonization.
        #   - Anderson (1991), continuous-time Markov chains.
        # Mathematical expressions:
        #   epsilon_fraction = |sum_i p_i(t + dt) - 1|.
        #   epsilon_delta = |sum_i Delta p_i|.
        # Parameter meanings:
        #   p_i: fraction of land-use class i [1].
        #   Delta p_i: per-step land-use fraction change [1].
        # Soil-carbon input-rate change is a diagnostic coupling observation;
        # it is not summed into the closed land-area-fraction ledger.
        return self._check_conservation_laws(input_state, output_updates)

    def _enforce_conservation_policy(self, conservation: Dict[str, float]) -> None:
        """Apply parameterized conservation tolerance and strictness."""
        tolerance_param = self.get_parameter_value(
            "conservation_tolerance",
            namespace="validation.conservation",
            source_order=("manager",), required=True)
        tolerance = float(tolerance_param)

        violations = {
            law: abs(float(value))
            for law, value in conservation.items()
            if abs(float(value)) > tolerance
        }
        if not violations:
            return

        message = (
            f"LandUseTransitions conservation violation exceeds tolerance {tolerance}: {violations}"
        )
        logger.warning(message)
        raise ValueError(message)
    
    def _apply_transition_matrix(
        self,
        fractions: torch.Tensor,
        dt: float,
        transition_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply transition matrix to land use fractions
        
        Args:
            fractions: Current land use fractions [n_types]
            dt: Time step [s]
            
        Returns:
            Change in fractions [n_types]
        """
        if dt <= 0.0:
            raise ValueError("Transition matrix application requires dt > 0.")
        # Convert seconds to years using the physical-constants SSOT.
        seconds_per_year = float(get_physical_constant("seconds_per_year"))
        dt_years = dt / seconds_per_year

        generator = transition_matrix if transition_matrix is not None else self.transition_matrix
        generator = generator.to(device=fractions.device, dtype=fractions.dtype)
        prob_matrix = torch.matrix_exp(generator * dt_years)
        # References: Hurtt et al. (2020); Anderson (1991).
        # Mathematical expression: p_new = p exp(Q dt_years).
        # Parameter meanings: p is the class fraction state [1], Q is the
        # ParameterManager-backed CTMC generator [yr^-1], and dt_years is [yr].
        type_axis = self._land_use_type_axis(fractions)
        class_last = torch.movedim(fractions, type_axis, -1)
        updated_class_last = torch.matmul(class_last, prob_matrix)
        updated = torch.movedim(updated_class_last, -1, type_axis)
        return updated - fractions
    
    def _apply_transition_events(self,
                                fractions: torch.Tensor,
                                time_index: int) -> torch.Tensor:
        """
        Apply discrete transition events
        
        Args:
            fractions: Current land use fractions [n_types]
            time_index: Current time index
            
        Returns:
            Change in fractions from events [n_types]
        """
        delta_fractions = torch.zeros_like(fractions)
        type_axis = self._land_use_type_axis(fractions)
        
        for event in self.transition_events:
            # Check if event should occur at this time
            if event.time_index is None or event.time_index == time_index:
                from_idx = self.land_use_types.index(event.from_type)
                to_idx = self.land_use_types.index(event.to_type)
                
                source_fraction = fractions.select(type_axis, from_idx)
                transition_amount = source_fraction * event.area_fraction

                # Apply spatial mask if provided
                if event.spatial_mask is not None:
                    mask = event.spatial_mask.to(
                        device=fractions.device,
                        dtype=fractions.dtype,
                    )
                    if mask.numel() != 1 and mask.shape != source_fraction.shape:
                        raise ValueError(
                            "event.spatial_mask must be scalar or match the spatial support"
                        )
                    transition_amount = transition_amount * mask

                source_index = [slice(None)] * fractions.dim()
                target_index = [slice(None)] * fractions.dim()
                source_index[type_axis] = from_idx
                target_index[type_axis] = to_idx
                delta_fractions[tuple(source_index)] -= transition_amount
                delta_fractions[tuple(target_index)] += transition_amount
        
        return delta_fractions
    
    def forward(
        # References: Hurtt et al. (2020); IPCC (2019); Verburg et al. (2009).
        # Mathematical expression: p_dot = p Q(theta); p(t + dt) = p exp(Q dt_years); Q 1 = 0;
        #   H_Q = diag(sum_{j != i} Q_ij) >= 0.
        # Parameter meanings: p is the land-use fraction row vector on the nine LUH2 classes;
        #   Q the transition-rate generator with zero row sums; dt_years the step in years;
        #   H_Q the declared positive semidefinite transition mobility.
        # Units: p [1]; Q [yr-1]; dt_years [yr]; outputs carry OUTPUT_UNITS.
        # Domain/constraints: exactly nine classes on the class axis, other axes are batch
        #   supports; zero row sums keep sum_i p_i = 1 and fractions stay within [0, 1].
        # Sign convention: Q_ij > 0 (i != j) moves fraction from class i to class j; diagonal
        #   entries are the negative total outflow.
        self,
        state: Dict[str, torch.Tensor],
        time: float = 0.0,
        dt: float = 1.0,
    ) -> Dict[str, UpdateItem]:
        """
        Advance the canonical land-use fraction state and emit observations.

        The canonical state is ``land_use_fraction``. The class axis contains
        exactly the nine declared LUH2 classes, and all other dimensions are
        preserved as independent batch or spatial supports.
        """
        self._scalar_value(time, "time")
        fractions, type_axis = self._validate_fraction_state(state)
        dt_value = self._scalar_value(dt, "dt")
        if dt_value < 0.0:
            raise ValueError("dt must be non-negative")

        if dt_value == 0.0:
            updated_fractions = fractions.clone()
        else:
            generator = self._compute_transition_matrix(state)
            delta_fraction = self._apply_transition_matrix(
                fractions,
                dt_value,
                transition_matrix=generator,
            )
            if "time_index" in state:
                time_index = int(state["time_index"].detach().reshape(-1)[0].item())
                delta_fraction = delta_fraction + self._apply_transition_events(
                    fractions,
                    time_index,
                )
            updated_fractions = fractions + delta_fraction

        if not torch.isfinite(updated_fractions).all():
            raise ValueError("Updated land_use_fraction contains non-finite values")
        if torch.any(updated_fractions < 0.0):
            raise ValueError("Land-use transitions produced a negative fraction")

        tolerance = float(
            self.get_parameter_value(
                "conservation_tolerance",
                namespace="validation.conservation",
                source_order=("manager",),
                required=True,
            )
        )
        totals = updated_fractions.sum(dim=type_axis)
        if torch.max(torch.abs(totals - 1.0)) > tolerance:
            raise ValueError("Land-use transitions violated area-fraction closure")

        outputs = self._build_runtime_outputs(
            fractions,
            updated_fractions,
            type_axis,
        )
        conservation = self._check_conservation_laws(state, outputs)
        self._enforce_conservation_policy(conservation)
        return outputs
    
    @staticmethod
    def _validate_scale_data(data: Tensor, *, role: str) -> None:
        """Validate the typed cochain payload used by a scale chain map."""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{role} data must be a torch.Tensor")
        if data.numel() == 0:
            raise ValueError(f"{role} data must not be empty")
        if not torch.isfinite(data).all():
            raise ValueError(f"{role} data must contain only finite values")

    def _apply_scale_chain_map(
        self,
        data: Tensor,
        *,
        bridge: Optional[Any],
        bridge_name: str,
    ) -> Tensor:
        """Apply one existing bound scale bridge and validate its typed output."""
        self._validate_scale_data(data, role="source")
        if bridge is None:
            raise RuntimeError(
                f"{bridge_name} requires an explicit typed chain-map bridge"
            )
        result = bridge(data)
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"{bridge_name} must return a torch.Tensor")
        if result.numel() == 0:
            raise RuntimeError(f"{bridge_name} returned an empty target cochain")
        if result.device != data.device:
            raise RuntimeError(
                f"{bridge_name} output must use the source cochain device"
            )
        if result.dtype != data.dtype:
            raise RuntimeError(
                f"{bridge_name} output must preserve the source cochain dtype"
            )
        if not torch.isfinite(result).all():
            raise ValueError(f"{bridge_name} output must contain only finite values")
        return result

    def upscale(self, fine_data: Tensor) -> Tensor:
        """Apply the explicitly bound fine-to-coarse typed chain map."""
        # References: Hurtt et al. (2020); Lawrence et al. (2016).
        # Mathematical expression: p_coarse = R_h p_fine.
        # Parameter meanings: R_h is the configured conservative chain map;
        # p_fine and p_coarse are land-use cochains on declared grid supports.
        return self._apply_scale_chain_map(
            fine_data,
            bridge=self._upscale_bridge,
            bridge_name="upscale",
        )
    
    def downscale(self, coarse_data: Tensor) -> Tensor:
        """Apply the explicitly bound coarse-to-fine typed chain map."""
        # References: Hurtt et al. (2020); Lawrence et al. (2016).
        # Mathematical expression: p_fine = P_h p_coarse.
        # Parameter meanings: P_h is the configured prolongation chain map;
        # p_coarse and p_fine are land-use cochains on declared grid supports.
        return self._apply_scale_chain_map(
            coarse_data,
            bridge=self._downscale_bridge,
            bridge_name="downscale",
        )
    
    def resolution_converter(self, 
                           data: Tensor,
                           source_resolution: float,
                           target_resolution: float,
                           method: str = "conservative") -> Tensor:
        """
        Convert land use transition data between different spatial resolutions.
        
        Provides resolution conversion for land use fractions and transition rates
        while preserving total area and maintaining fractional constraints.
        
        Parameters:
            data : Tensor
                Land use data to convert between resolutions
                Shape: [n_landuse_types] or [spatial_shape]
                Units: [fraction] or [yr^-1]
            source_resolution : float
                Source grid resolution [m]
            target_resolution : float
                Target grid resolution [m]
            method : str
                Conversion method ("conservative", "bilinear", "fraction_weighted")
                Default: "conservative"
                
        Returns:
            Tensor: Resolution-converted land use data
                Shape depends on resolution ratio and method
                Units: [fraction] or [yr^-1]
                
        Mathematical Expression:
            Land use fraction conservation: sum_types f_type = 1
            Area conservation: integral f_type(x) dA = constant for each type
            
            For fraction data:
            f_target = weighted_average(f_source, area_weights)
            
            For transition rates:
            T_target = area_weighted_average(T_source)
        """
        self._validate_scale_data(data, role="source")
        if source_resolution <= 0 or target_resolution <= 0:
            raise ValueError("Resolutions must be positive")
        supported_methods = {"conservative", "bilinear", "fraction_weighted"}
        if method not in supported_methods:
            raise ValueError(
                f"Unsupported resolution conversion method: {method}"
            )

        resolution_ratio = target_resolution / source_resolution

        if abs(resolution_ratio - 1.0) < 1e-6:
            # Identity is exact because source and target supports coincide.
            return data

        # The selected method is implemented by the bound bridge owner. This
        # functor only dispatches by scale direction and never invents topology.
        if resolution_ratio > 1.0:
            return self.upscale(data)
        return self.downscale(data)


    
