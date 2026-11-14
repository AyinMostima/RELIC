"""Canonical resistance-law dry deposition transport.

References:
    Wesely, M. L. (1989), Atmos. Environ., 23, 1293-1304.
    Zhang, L., Brook, J. R., and Vet, R. (2003), ACP, 3, 2067-2082.

Mathematical expression:
    r_eff,s = r_b,s + r_c,s
    v_d,s = 1 / (r_a + r_eff,s)
    dC_s/dt = -v_d,s C_s / h_pbl
    Delta C_s = max(-C_s, dC_s/dt * Delta t)

Parameter meanings and units:
    C_s is species mixing ratio [mol mol-1].
    r_a is upstream aerodynamic resistance [s m-1].
    r_eff,s is the ParameterManager-backed effective non-aerodynamic
    resistance r_b,s + r_c,s [s m-1].
    h_pbl is planetary boundary-layer height [m].
    v_d,s is dry-deposition velocity [m s-1].

Open-system route:
    Negative atmospheric increments leave through the declared
    dry_deposition_surface_export boundary port. The availability limiter
    residual is reported explicitly as a diagnostic.
"""
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import torch
from torch import Tensor
from ....computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ....computation.operators.iphs import PortSpecIPHS
from ....core.functor import (
    TransportMechanismFunctor,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from ....core.abstractions import StateDict, FluxDict
from ....core.state import UpdateItem, UpdateMode
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SpeciesDepositionParams:
    """ParameterManager-backed effective resistance r_b + r_c [s m-1]."""

    effective_resistance: float

@dataclass(frozen=True)
class DryDepositionConfig:
    """Structural activation configuration; numerical values remain in YAML."""

    enabled: Optional[bool] = None

class DryDepositionAggregator(TransportMechanismFunctor):
    """Apply the single declared resistance-law dry-deposition closure."""

    TRANSPORT_TYPE = "dry_deposition"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    _CANONICAL_SPECIES = ('no', 'no2', 'o3', 'so2')
    SURFACE_LAYER_SOURCES = ('atmospheric_turbulence', 'external_forcing')

    def __init__(
        self,
        name: str = 'dry_deposition_aggregator',
        config: Optional[Dict[str, Any]] = None,
        deposition_config: Optional[DryDepositionConfig] = None,
        *,
        surface_layer_source: Literal[
            'atmospheric_turbulence', 'external_forcing'
        ] = 'atmospheric_turbulence',
    ) -> None:
        """Initialize with structural activation only; numerics come from YAML.

        ``surface_layer_source`` is a GraphIR selection field, not a parameter. It names
        the provider of aerodynamic_resistance: the atmospheric_turbulence Functor, a
        required dependency, or external forcing such as an observed tower series.
        The law evaluated in ``forward`` is the same for both.
        """

        if surface_layer_source not in self.SURFACE_LAYER_SOURCES:
            raise ValueError(
                f'surface_layer_source must be one of {self.SURFACE_LAYER_SOURCES}, '
                f'got {surface_layer_source!r}.'
            )
        if config is None:
            config = (
                {'dependencies': ['atmospheric_turbulence']}
                if surface_layer_source == 'atmospheric_turbulence'
                else {}
            )
        elif 'parameters' in config:
            raise ValueError(
                'Dry deposition numerical overrides are not allowed; use ParameterManager YAML.'
            )
        metadata = FunctorMetadata(
            name=name,
            version='2.0.0',
            description='Canonical resistance-law atmospheric dry deposition',
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                '$r_{eff,s}=r_{b,s}+r_{c,s}$; '
                '$v_{d,s}=(r_a+r_{eff,s})^{-1}$; '
                '$\\Delta C_s=\\max(-C_s,-v_{d,s}C_s\\Delta t/h_{pbl})$'
            ),
            references=[
                'Wesely, M.L. (1989). Atmos. Environ., 23, 1293-1304',
                'Zhang, L., Brook, J.R., & Vet, R. (2003). Atmos. Chem. Phys., 3, 2067-2082',
            ],
            parameters={
                'atmosphere.dry_deposition.enabled': (
                    'Activation switch of the dry deposition aggregator [dimensionless]'
                ),
                'atmosphere.dry_deposition.species_list': (
                    'Ordered canonical deposited species [dimensionless]'
                ),
                'atmosphere.dry_deposition.species.no.effective_non_aerodynamic_resistance': (
                    'Effective resistance r_b + r_c of NO [s m-1]'
                ),
                'atmosphere.dry_deposition.species.no2.effective_non_aerodynamic_resistance': (
                    'Effective resistance r_b + r_c of NO2 [s m-1]'
                ),
                'atmosphere.dry_deposition.species.o3.effective_non_aerodynamic_resistance': (
                    'Effective resistance r_b + r_c of O3 [s m-1]'
                ),
                'atmosphere.dry_deposition.species.so2.effective_non_aerodynamic_resistance': (
                    'Effective resistance r_b + r_c of SO2 [s m-1]'
                ),
                'atmosphere.dry_deposition.reference_height': (
                    'Reference height of the deposition velocity [m]'
                ),
                'atmosphere.dry_deposition.structural_reference_deposition_velocity': (
                    'Reference deposition velocity of the static Hodge contract [m s-1]'
                ),
            },
            input_variables=[
                'no_mixing_ratio',
                'no2_mixing_ratio',
                'o3_mixing_ratio',
                'so2_mixing_ratio',
                'pbl_height',
                'aerodynamic_resistance',
            ],
            output_variables=[
                'no_mixing_ratio',
                'no2_mixing_ratio',
                'o3_mixing_ratio',
                'so2_mixing_ratio',
                'diagnostics.dry_deposition.total_mixing_ratio_increment',
                'diagnostics.dry_deposition.total_limiter_residual',
                'diagnostics.dry_deposition.normalized_mass_balance_residual',
                'diagnostics.dry_deposition.enabled',
            ],
            conservation_laws=['mass'],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=['atmospheric_turbulence'],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
        )
        self.surface_layer_source = surface_layer_source
        self.config_deposition = self._load_deposition_config(deposition_config)
        self.target_species = self._load_target_species()
        self.species_parameters = MappingProxyType(
            self._load_species_params(self.target_species)
        )
        self.species_map = MappingProxyType(
            {species: f'{species}_mixing_ratio' for species in self.target_species}
        )
        self._structural_parameters = MappingProxyType(
            {
                'reference_height': self._parameter_scalar('reference_height'),
                'reference_velocity': self._parameter_scalar(
                    'structural_reference_deposition_velocity'
                ),
            }
        )
        # Dry deposition has no constant-diffusivity law: the inherited scalar
        # diffusivity stays unset (None).
        logger.info(
            "Initialized %s '%s' (enabled=%s, closure=resistance_law)",
            self.__class__.__name__,
            name,
            self.config_deposition.enabled,
        )

    @staticmethod
    def _scalar_from_value(value: Any) -> float:
        """Convert scalar tensor or numeric value to Python float without host sync calls."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Expected scalar tensor value.")
            detached = value.detach().reshape(-1)
            values = detached.tolist()
            if not values:
                raise ValueError("Expected non-empty scalar tensor value.")
            return float(values[0])
        return float(value)

    def _parameter_scalar(self, name: str, namespace: str = 'atmosphere.dry_deposition') -> float:
        """Load one finite scalar from the Functor parameter path."""

        value = self.get_parameter_value(name, namespace=namespace, required=True)
        scalar = self._scalar_from_value(value)
        if not torch.isfinite(torch.tensor(scalar)):
            raise ValueError(f'Dry deposition parameter {namespace}.{name} must be finite.')
        return scalar

    def _load_deposition_config(
        self,
        deposition_config: Optional[DryDepositionConfig],
    ) -> DryDepositionConfig:
        """Resolve only the structural activation flag."""

        base_ns = 'atmosphere.dry_deposition'
        if deposition_config is not None and deposition_config.enabled is not None:
            if type(deposition_config.enabled) is not bool:
                raise TypeError('Dry deposition enabled must be bool when provided.')
            enabled = deposition_config.enabled
        else:
            value = self.get_parameter_value('enabled', namespace=base_ns, required=True)
            if type(value) is not bool:
                raise TypeError('ParameterManager dry deposition enabled must be bool.')
            enabled = value
        return DryDepositionConfig(enabled=enabled)

    def _load_target_species(self) -> tuple[str, ...]:
        """Load and validate the exact canonical species ordering."""

        raw = self.get_parameter_value(
            'species_list',
            namespace='atmosphere.dry_deposition',
            required=True,
        )
        if not isinstance(raw, (list, tuple)) or isinstance(raw, (str, bytes)):
            raise TypeError('Dry deposition species_list must be an ordered list or tuple.')
        species = tuple(str(item) for item in raw)
        if species != self._CANONICAL_SPECIES:
            raise ValueError(
                'Dry deposition species_list must equal '
                f'{self._CANONICAL_SPECIES}; received {species}.'
            )
        return species

    def _load_species_params(
        self,
        species_list: tuple[str, ...],
    ) -> Dict[str, SpeciesDepositionParams]:
        """Load positive species surface resistances from ParameterManager."""

        params: Dict[str, SpeciesDepositionParams] = {}
        for species in species_list:
            namespace = f'atmosphere.dry_deposition.species.{species}'
            resistance = self._parameter_scalar(
                'effective_non_aerodynamic_resistance',
                namespace=namespace,
            )
            if resistance <= 0.0:
                raise ValueError(
                    f'Dry deposition effective resistance for {species} must be positive.'
                )
            params[species] = SpeciesDepositionParams(
                effective_resistance=resistance
            )
        return params

    def get_dependencies(self) -> List[str]:
        """Declare the sole producing dependency for aerodynamic resistance.

        With surface_layer_source = external_forcing the value is forcing and no
        producer exists; a missing value raises in forward. Instance declarations
        are kept in both cases.
        """

        if self.surface_layer_source == 'external_forcing':
            return self._merge_declared_dependencies([])
        return self._merge_declared_dependencies(['atmospheric_turbulence'])

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare dry deposition cochain spaces for Phase 8 audit.

        References:
          - Wesely (1989), resistance analogy for gaseous dry deposition.
          - Zhang et al. (2003), revised gaseous dry deposition parameterization.
        Mathematical expression:
          x_dot = -D_h q_h + G_h u_h, q_h = star_dep C_h.
        Parameter meanings:
          C_h is the near-surface species mixing-ratio cochain, star_dep is the
          non-negative deposition mobility, and q_h is the surface sink flow.
        """

        n_species = len(self.target_species)
        return (
            CochainSpaceSpec(
                complex_id="atmosphere_dry_deposition",
                degree=0,
                primality="primal",
                n_dofs=n_species,
                units="mol mol-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="near_surface_species_mixing_ratio",
                orientation_convention="upward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmosphere_dry_deposition",
                degree=2,
                primality="dual",
                n_dofs=n_species,
                units="mol mol-1 s-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="dry_deposition_normalized_boundary_sink",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "atmosphere.dry_deposition.enabled",
            "atmosphere.dry_deposition.reference_height",
            "atmosphere.dry_deposition.structural_reference_deposition_velocity",
            "atmosphere.dry_deposition.species_list",
            *tuple(
                'atmosphere.dry_deposition.species.'
                f'{species}.effective_non_aerodynamic_resistance'
                for species in self.target_species
            ),
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive dry-deposition mobility Hodge contract.

        References:
          - Wesely (1989), dry deposition resistance analogy.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = star_dep C_h, star_dep = v_d / h_ref >= 0.
        Parameter meanings:
          v_d is deposition velocity and h_ref is the reference or mixing height.
        """

        n_local = int(n_dofs) if n_dofs is not None else len(self.target_species)
        if n_local < 1:
            raise ValueError("Dry deposition Hodge degrees of freedom must be positive.")
        reference_height = self._structural_parameters['reference_height']
        if reference_height <= 0.0:
            raise ValueError("Dry deposition reference height must be positive.")
        mobility = (
            self._structural_parameters['reference_velocity']
            / reference_height
        )
        if mobility <= 0.0:
            raise ValueError('Dry deposition structural mobility must be positive.')
        values = torch.full(
            (n_local,),
            mobility,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        concentration_space = CochainSpaceSpec(
            complex_id="atmosphere_dry_deposition",
            degree=0,
            primality="primal",
            n_dofs=n_local,
            units="mol mol-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="near_surface_species_mixing_ratio",
            orientation_convention="upward_positive",
        )
        sink_space = CochainSpaceSpec(
            complex_id="atmosphere_dry_deposition",
            degree=2,
            primality="dual",
            n_dofs=n_local,
            units="mol mol-1 s-1",
                extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="dry_deposition_normalized_boundary_sink",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_dry_deposition_hodge",
            domain=concentration_space,
            codomain=sink_space,
            matrix=torch.diag(values),
            parameter_keys=(
                "atmosphere.dry_deposition.structural_reference_deposition_velocity",
                "atmosphere.dry_deposition.reference_height",
            ),
            metric_keys=("atmosphere.dry_deposition.reference_height",),
            material_law="dry_deposition_surface_mobility",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build dry-deposition Hodge terms for Phase 8 assembly."""

        n_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural concentration audit view."""

        vectors: List[Tensor] = []
        for state_var in self.species_map.values():
            if state_var in state:
                vectors.append(self._ensure_tensor(state[state_var]).reshape(-1))
        if not vectors:
            return {
                "quadratic_storage_audit": "H_audit = 0.5 * C_h.T C_h",
                "effort_audit": "e_h = C_h",
            }
        # References: Phase 8 structural audit convention.
        # Mathematical expression: H_audit = 0.5 * C_h.T C_h.
        # Parameter meanings: C_h is the near-surface species mixing-ratio cochain.
        concentration = torch.cat(vectors)
        return {
            "quadratic_storage_audit": 0.5 * torch.sum(concentration * concentration),
            "effort_audit": concentration,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative structural dry-deposition dissipation terms."""

        vectors: List[Tensor] = []
        for state_var in self.species_map.values():
            if state_var in state:
                vectors.append(torch.clamp(self._ensure_tensor(state[state_var]).reshape(-1), min=0.0))
        if not vectors:
            return {
                "dissipation_rate": "sigma_h = C_h.T star_dep C_h",
                "thermodynamic_force": "C_h",
            }
        concentration = torch.cat(vectors)
        hodge = self.build_hodge_contract(n_dofs=concentration.numel(), strict=True).matrix
        # References: Wesely (1989); port-Hamiltonian dissipative transport form.
        # Mathematical expression: sigma_h = C_h.T star_dep C_h >= 0.
        # Parameter meanings: star_dep is the non-negative deposition mobility.
        return {
            "dissipation_rate": concentration @ hodge @ concentration,
            "thermodynamic_force": concentration,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for dry deposition."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive dry-deposition sink mobility."""

        n_dofs = len(self.target_species)
        hodge = self.build_hodge_contract(n_dofs=n_dofs, strict=True).matrix
        if bool(torch.any(hodge < 0.0)):
            raise ValueError("Dry deposition mobility must be non-negative in M_h.")
        return {
            "state_vars": tuple(self.species_map[species] for species in self.target_species),
            "J": torch.zeros_like(hodge),
            "M": hodge,
            "ports": self.ports(),
            "open_system_route": "boundary_export",
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the dry-deposition surface export port.

        References:
          - Wesely (1989), surface resistance dry deposition.
          - Zhang et al. (2003), dry-deposition velocity framework.
        Mathematical expression:
          P_port = e_h q_dep, q_dep = -v_d C_h / h_ref.
        Parameter meanings:
          e_h is the near-surface concentration effort and q_dep is the surface sink.
        """

        effort_space = CochainSpaceSpec(
            complex_id="atmosphere_dry_deposition",
            degree=0,
            primality="primal",
            n_dofs=len(self.target_species),
            units="mol mol-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="near_surface_species_mixing_ratio",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="atmosphere_dry_deposition",
            degree=2,
            primality="dual",
            n_dofs=len(self.target_species),
            units="mol mol-1 s-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="dry_deposition_normalized_boundary_sink",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="dry_deposition_surface_export",
                effort_var="near_surface_species_mixing_ratio",
                flow_var="dry_deposition_mixing_ratio_tendency",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="mol2 mol-2 s-1",
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

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """Return direct atmospheric state increments from the sole closure."""

        # References:
        # - Wesely (1989); Zhang et al. (2003).
        # Mathematical expression:
        # - r_eff,s = r_b,s + r_c,s
        # - v_d,s = 1 / (r_a + r_eff,s)
        # - Delta C_s = max(-C_s, -v_d,s C_s dt / h_pbl)
        # Parameter meanings:
        # - r_a: upstream aerodynamic resistance [s m-1].
        # - r_eff,s: effective non-aerodynamic resistance [s m-1].
        # - h_pbl: boundary-layer height [m].
        # - C_s: atmospheric mixing ratio [mol mol-1].
        # Units:
        # - Delta C_s: [mol mol-1] for the current timestep.
        # Open-system route:
        # - The negative increment exits through dry_deposition_surface_export.
        if not self.config_deposition.enabled:
            return self._create_empty_output()

        required = [
            *(self.species_map[species] for species in self.target_species),
            'pbl_height',
            'aerodynamic_resistance',
        ]
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f'Dry deposition missing required state variables: {missing}')

        first_state = self._require_state_tensor(
            state,
            self.species_map[self.target_species[0]],
        )
        dt_tensor = torch.as_tensor(
            dt,
            device=first_state.device,
            dtype=first_state.dtype,
        )
        if dt_tensor.numel() != 1 or not bool(torch.isfinite(dt_tensor).all()):
            raise ValueError('Dry deposition dt must be one finite scalar.')
        if bool(dt_tensor <= 0.0):
            raise ValueError('Dry deposition dt must be positive.')

        pbl_height = self._require_state_tensor(state, 'pbl_height')
        aerodynamic_resistance = self._require_state_tensor(
            state,
            'aerodynamic_resistance',
        )
        self._require_positive_finite(pbl_height, 'pbl_height')
        self._require_positive_finite(
            aerodynamic_resistance,
            'aerodynamic_resistance',
        )

        updates: Dict[str, Tensor] = {}
        residuals: List[Tensor] = []
        balance_residuals: List[Tensor] = []
        for species in self.target_species:
            state_key = self.species_map[species]
            mixing_ratio = self._require_state_tensor(state, state_key)
            if bool(torch.any(~torch.isfinite(mixing_ratio))):
                raise ValueError(f'{state_key} must be finite.')
            if bool(torch.any(mixing_ratio < 0.0)):
                raise ValueError(f'{state_key} must be non-negative.')
            resistance = mixing_ratio.new_tensor(
                self.species_parameters[species].effective_resistance
            )
            ra, height, concentration = torch.broadcast_tensors(
                aerodynamic_resistance.to(
                    device=mixing_ratio.device,
                    dtype=mixing_ratio.dtype,
                ),
                pbl_height.to(
                    device=mixing_ratio.device,
                    dtype=mixing_ratio.dtype,
                ),
                mixing_ratio,
            )
            deposition_velocity = torch.reciprocal(ra + resistance)
            raw_increment = (
                -deposition_velocity * concentration * dt_tensor / height
            )
            increment = torch.maximum(raw_increment, -concentration)
            increment = torch.minimum(
                increment,
                torch.zeros_like(increment),
            )
            updates[state_key] = increment
            limiter_residual = increment - raw_increment
            residuals.append(limiter_residual)
            balance_residuals.append(
                increment - raw_increment - limiter_residual
            )

        update_values = torch.broadcast_tensors(*updates.values())
        residual_values = torch.broadcast_tensors(*residuals)
        balance_residual_values = torch.broadcast_tensors(*balance_residuals)
        total_increment = torch.stack(update_values, dim=0).sum(dim=0)
        total_residual = torch.stack(residual_values, dim=0).sum(dim=0)
        normalized_mass_balance_residual = torch.stack(
            balance_residual_values,
            dim=0,
        )
        output: FluxDict = {
            'no_mixing_ratio': UpdateItem(
                value=updates['no_mixing_ratio'],
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'no2_mixing_ratio': UpdateItem(
                value=updates['no2_mixing_ratio'],
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'o3_mixing_ratio': UpdateItem(
                value=updates['o3_mixing_ratio'],
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'so2_mixing_ratio': UpdateItem(
                value=updates['so2_mixing_ratio'],
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
        }
        output[
            'diagnostics.dry_deposition.total_mixing_ratio_increment'
        ] = total_increment
        output[
            'diagnostics.dry_deposition.total_limiter_residual'
        ] = total_residual
        output[
            'diagnostics.dry_deposition.normalized_mass_balance_residual'
        ] = normalized_mass_balance_residual
        output[
            'diagnostics.dry_deposition.enabled'
        ] = total_increment.new_ones(())
        return output

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the per-species normalized deposition boundary ledger.

        References:
          - Wesely (1989), resistance-in-series dry deposition.
          - Zhang et al. (2003), species-dependent deposition velocity.
        Mathematical expression:
          P_s = v_d,s C_s dt / h_pbl, E_s = min(P_s, C_s),
          L_s = P_s - E_s, and R_s = Delta C_s + P_s - L_s = 0.
        Parameter meanings:
          P_s is potential normalized export, E_s is retained export, L_s is
          availability-denied export, and Delta C_s = -E_s; all use mol mol-1.
        Open-system route:
          Each species leaves the well-mixed boundary layer through the
          dry-deposition boundary. This normalized audit does not claim a
          physical surface molar flux without air molar density.
        """

        del input_state
        residual_key = (
            'diagnostics.dry_deposition.normalized_mass_balance_residual'
        )
        if residual_key not in output_updates:
            raise KeyError(f"Missing conservation output '{residual_key}'")
        residual = output_updates[residual_key]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError(
                'Dry-deposition normalized mass residual must be a torch.Tensor.'
            )
        if residual.numel() == 0:
            raise ValueError(
                'Dry-deposition normalized mass residual must be non-empty.'
            )
        if not bool(torch.all(torch.isfinite(residual))):
            raise ValueError(
                'Dry-deposition normalized mass residual must be finite.'
            )
        magnitude = torch.max(torch.abs(residual))
        return {'mass': float(magnitude.detach().item())}

    @staticmethod
    def _require_state_tensor(state: StateDict, key: str) -> Tensor:
        """Return one declared state tensor without compatibility coercion."""

        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f'Dry deposition state {key} must be torch.Tensor.')
        return value

    @staticmethod
    def _require_positive_finite(value: Tensor, name: str) -> None:
        """Fail fast for non-finite or non-positive resistance/height inputs."""

        if bool(torch.any(~torch.isfinite(value))):
            raise ValueError(f'Dry deposition input {name} must be finite.')
        if bool(torch.any(value <= 0.0)):
            raise ValueError(f'Dry deposition input {name} must be positive.')

    def _create_empty_output(self) -> FluxDict:
        """Return only an explicit disabled diagnostic and no state writer."""

        return {
            'diagnostics.dry_deposition.normalized_mass_balance_residual': (
                torch.zeros(
                    (len(self._CANONICAL_SPECIES),),
                    device=self.device,
                )
            ),
            'diagnostics.dry_deposition.enabled': torch.zeros(
                (),
                device=self.device,
            )
        }

    def _ensure_tensor(self, value: Union[float, int, Tensor]) -> Tensor:
        """Ensure a structural-audit value is a tensor on the Functor device."""

        if isinstance(value, Tensor):
            return value.to(self.device)
        return torch.tensor(float(value), device=self.device)

    def _resolve_diffusivity_tensor(self, field_values: Tensor) -> Tensor:
        """Reject the inherited constant-diffusivity execution path."""

        raise RuntimeError(
            'Dry deposition is a surface resistance process; use forward() with '
            'aerodynamic_resistance and pbl_height.'
        )

    def boundary_conditions(
        self,
        boundary_name: str,
        field_values: Tensor,
        normal_vector: Tensor,
        boundary_type: str,
        boundary_value: Optional[Tensor] = None,
    ) -> Tensor:
        """Reject the unrelated inherited diffusion boundary path."""

        raise RuntimeError(
            'Dry deposition boundary export is evaluated only by forward().'
        )

    def compute_diffusivity_tensor(
        self,
        temperature: Tensor,
        pressure: Tensor,
        medium_properties: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Reject the unrelated inherited bulk diffusivity path."""

        raise RuntimeError(
            'Dry deposition has no bulk diffusivity tensor; use forward().'
        )

    def get_transport_metadata(self) -> Dict[str, Any]:
        """Return metadata for the sole canonical resistance closure."""

        return {
            'transport_type': self.transport_type,
            'closure': 'resistance_law',
            'surface_layer_source': self.surface_layer_source,
            'enabled': self.config_deposition.enabled,
            'target_species': self.target_species,
            'species_effective_non_aerodynamic_resistance': {
                species: params.effective_resistance
                for species, params in self.species_parameters.items()
            },
            'required_state_inputs': (
                'pbl_height',
                'aerodynamic_resistance',
            ),
            'open_system_route': 'boundary_export',
            'strict_surface_flux_proof': False,
            'normalization_assumption': (
                'well_mixed_pbl_with_constant_air_molar_density'
            ),
        }

    def get_deposition_metadata(self) -> Dict[str, Any]:
        """Return dry-deposition-specific canonical metadata."""

        return self.get_transport_metadata()

    def __repr__(self) -> str:
        """Return a concise canonical closure description."""

        enabled_str = 'enabled' if self.config_deposition.enabled else 'disabled'
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"closure='resistance_law', {enabled_str}, "
            f"species={len(self.target_species)})"
        )


__all__ = ['DryDepositionAggregator', 'DryDepositionConfig']
