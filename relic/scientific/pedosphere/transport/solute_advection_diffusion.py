"""Conservative one-dimensional transport of dissolved NO3 and DOC.

References
----------
Simunek et al. (2008), Vadose Zone Journal 7, 587-600; HYDRUS-1D 4.16
technical manual; Bear (1972), Dynamics of Fluids in Porous Media.

Mathematical expression
-----------------------
For aqueous concentration C and linear equilibrium sorption, the stored mass
per bulk volume is B C with B = theta + rho_b K_d. The conservative equation is

    partial(B C)/partial(t) = -partial(J)/partial(z)
    J = q C_upwind - K partial(C)/partial(z)
    K = alpha_L abs(q) + theta tau D_w(T)

where K_d is converted from L kg-1 to m3 kg-1 before use. The top Cauchy
boundary prescribes J_top = q_top C_surface. The free-drainage bottom boundary
has zero dispersive gradient and J_bottom = q_bottom C_bottom.

Parameter meanings and units
----------------------------
C is aqueous concentration [g element m-3], theta is volumetric water content
[m3 m-3], rho_b is bulk density [kg m-3], K_d is distribution coefficient
[m3 kg-1], q is downward Darcy flux [m s-1], alpha_L is longitudinal
dispersivity [m], tau is the Millington-Quirk tortuosity factor [-], D_w is
molecular diffusivity in water [m2 s-1], and z is depth [m]. Surface input and
bottom leaching are explicit open-system matter ports [g element m-2 s-1].
"""

from __future__ import annotations

import torch
from typing import Any, Dict, List, Mapping, Tuple, Optional
import logging

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TransportMechanismFunctor,
    VariableKind,
)
from relic.core.abstractions import StateDict
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.api_contract import (
    PedosphereContractMixin,
    FluxSpecification
)
from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS

from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


class SoluteAdvectionDiffusion1D(PedosphereContractMixin, TransportMechanismFunctor):
    """Finite-volume transport closure for aqueous NO3 and DOC."""

    TRANSPORT_TYPE = "solute_diffusion"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        n_layers: Optional[int] = None,
        soil_depth: Optional[float] = None,
        layer_thickness_ratio: Optional[float] = None,
        cfl_number: Optional[float] = None,
        max_timestep: Optional[float] = None,
        enable_sorption: bool = True,
    ):
        """Initialize the canonical two-species transport closure."""

        override_keys = []
        if n_layers is not None:
            override_keys.append("pedosphere.grid.n_layers")
        if soil_depth is not None:
            override_keys.append("pedosphere.grid.soil_depth")
        if layer_thickness_ratio is not None:
            override_keys.append("pedosphere.transport.layer_thickness_ratio")
        if cfl_number is not None:
            override_keys.append("pedosphere.transport.cfl_number")
        if max_timestep is not None:
            override_keys.append("pedosphere.transport.time_step_maximum")
        if override_keys:
            logger.warning(
                "Constructor overrides detected for %s; prefer ParameterManager values for persistence.",
                ", ".join(override_keys),
            )

        # Initialize metadata
        metadata = FunctorMetadata(
            name="SoluteAdvectionDiffusion1D",
            version="1.1.0",
            description="Conservative one-dimensional NO3 and DOC advection-dispersion transport",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                r"$\partial_t(B C)=-\partial_z J,\quad "
                r"J=q C_{upwind}-K\partial_z C,\quad "
                r"B=\theta+\rho_b K_d$"
            ),
            references=[
                "Bear, J. (1972). Dynamics of Fluids in Porous Media",
                "Simunek, J. et al. (2008). Vadose Zone Journal 7:587-600",
                "HYDRUS-1D Version 4.16 Technical Manual",
            ],
            parameters={
                "pedosphere.grid.n_layers": "Number of layers [-]",
                "pedosphere.grid.soil_depth": "Total soil depth [m]",
                "pedosphere.transport.cfl_number": "CFL number [-]",
                "pedosphere.transport.layer_thickness_ratio": "Geometric layer thickness ratio [-]",
                "pedosphere.transport.time_step_maximum": "Maximum time step [s]",
                "pedosphere.transport.default_molecular_diffusivity_water": "Reference structural diffusivity [m2 s-1]",
                "pedosphere.transport.nitrate_distribution_coefficient": "NO3 distribution coefficient [L kg-1]",
                "pedosphere.transport.doc_distribution_coefficient": "DOC distribution coefficient [L kg-1]",
                "pedosphere.transport.nitrate_diffusivity_water": "NO3 molecular diffusivity [m2 s-1]",
                "pedosphere.transport.doc_diffusivity_water": "DOC molecular diffusivity [m2 s-1]",
                "pedosphere.transport.nitrate_longitudinal_dispersivity": "NO3 longitudinal dispersivity [m]",
                "pedosphere.transport.doc_longitudinal_dispersivity": "DOC longitudinal dispersivity [m]",
                "pedosphere.transport.molecular_diffusivity_temperature_exponent": "Temperature exponent for solute molecular diffusivity scaling [-]",
                "pedosphere.physical.millington_quirk_exponent_numerator": "Millington-Quirk numerator exponent [-]",
                "pedosphere.physical.millington_quirk_exponent_denominator": "Millington-Quirk denominator exponent [-]",
            },
            input_variables=[
                "soil_water_content",
                "darcy_flux",
                "no3_concentration",
                "doc_concentration",
                "bulk_density",
                "porosity",
                "soil_temperature",
                "surface_no3_concentration",
                "surface_doc_concentration",
            ],
            output_variables=[
                "nitrogen_leaching_flux",
                "carbon_leaching_flux",
                "no3_concentration",
                "doc_concentration",
                "diagnostics.solute_transport.nitrogen_surface_input_flux",
                "diagnostics.solute_transport.carbon_surface_input_flux",
                "diagnostics.solute_transport.nitrogen_mass_balance_residual",
                "diagnostics.solute_transport.carbon_mass_balance_residual",
            ],
            conservation_laws=["mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            spatial_dimensions=1
        )
        
        # Initialize first so parameter access goes through FunctorBase/ParameterManager chain.
        super().__init__(
            name="SoluteAdvectionDiffusion1D",
            config={"parameters": {"enable_sorption": enable_sorption}},
            metadata=metadata,
        )

        def _as_float(value: Any) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.detach())
            return float(value)

        layer_thickness_ratio_value = _as_float(layer_thickness_ratio) if layer_thickness_ratio is not None else _as_float(
            self.get_parameter_value("layer_thickness_ratio", namespace="pedosphere.transport", required=True)
        )
        n_layers_value = int(n_layers) if n_layers is not None else int(
            self.get_parameter_value("n_layers", namespace="pedosphere.grid", required=True)
        )
        soil_depth_value = _as_float(soil_depth) if soil_depth is not None else _as_float(
            self.get_parameter_value("soil_depth", namespace="pedosphere.grid", required=True)
        )
        cfl_value = _as_float(cfl_number) if cfl_number is not None else _as_float(
            self.get_parameter_value("cfl_number", namespace="pedosphere.transport", required=True)
        )
        max_timestep_value = _as_float(max_timestep) if max_timestep is not None else _as_float(
            self.get_parameter_value("time_step_maximum", namespace="pedosphere.transport", required=True)
        )
        default_diffusivity_value = _as_float(
            self.get_parameter_value(
                "default_molecular_diffusivity_water",
                namespace="pedosphere.transport", required=True)
        )
        self.set_scalar_diffusivity(default_diffusivity_value)
        
        # Grid configuration
        self.n_layers = n_layers_value
        self.soil_depth = soil_depth_value
        self.cfl_number = cfl_value
        self.max_timestep = max_timestep_value
        self.enable_sorption = enable_sorption

        if self.n_layers < 1:
            raise ValueError("n_layers must be at least one.")
        if self.soil_depth <= 0.0:
            raise ValueError("soil_depth must be positive.")
        if not 0.0 < self.cfl_number <= 1.0:
            raise ValueError("cfl_number must be in (0, 1].")
        if self.max_timestep <= 0.0:
            raise ValueError("max_timestep must be positive.")
        if layer_thickness_ratio_value <= 0.0:
            raise ValueError("layer_thickness_ratio must be positive.")
        
        # Setup vertical grid
        self._setup_vertical_grid(layer_thickness_ratio_value)
        
        self._load_transport_parameters()
        
        logger.info(f"Initialized {self.metadata.name} with {self.n_layers} layers")
    
    def _setup_vertical_grid(self, thickness_ratio: float) -> None:
        """Register a nonuniform finite-volume grid as module buffers."""

        device = self.device if self.device is not None else torch.device("cpu")
        if thickness_ratio == 1.0:
            thickness = torch.full(
                (self.n_layers,),
                self.soil_depth / self.n_layers,
                dtype=torch.float32,
                device=device,
            )
        else:
            total_ratio = (1 - thickness_ratio**self.n_layers) / (1 - thickness_ratio)
            base_thickness = self.soil_depth / total_ratio
            indices = torch.arange(self.n_layers, dtype=torch.float32, device=device)
            thickness = base_thickness * torch.pow(
                torch.as_tensor(thickness_ratio, dtype=torch.float32, device=device),
                indices,
            )

        interfaces = torch.cat(
            (
                torch.zeros(1, dtype=thickness.dtype, device=device),
                torch.cumsum(thickness, dim=0),
            )
        )
        centers = interfaces[:-1] + 0.5 * thickness
        self.register_buffer("layer_thickness", thickness)
        self.register_buffer("layer_interfaces", interfaces)
        self.register_buffer("layer_centers", centers)

    def _fetch_default(self, namespace: str, name: str) -> float:
        """Helper to fetch scalar defaults from the ParameterManager."""
        value = self.get_parameter_value(name, namespace=namespace, required=True)
        if isinstance(value, torch.Tensor):
            return float(value.detach())
        return float(value)

    def _load_transport_parameters(self) -> None:
        """Load the parameters used by the single production transport law."""

        transport_ns = "pedosphere.transport"
        self._doc_dispersivity = self._fetch_default(transport_ns, "doc_longitudinal_dispersivity")
        self._nitrate_dispersivity = self._fetch_default(transport_ns, "nitrate_longitudinal_dispersivity")
        self._doc_distribution_coefficient = self._fetch_default(
            transport_ns,
            "doc_distribution_coefficient",
        )
        self._nitrate_distribution_coefficient = self._fetch_default(
            transport_ns,
            "nitrate_distribution_coefficient",
        )
        self._doc_diffusivity_water = self._fetch_default(
            transport_ns,
            "doc_diffusivity_water",
        )
        self._nitrate_diffusivity_water = self._fetch_default(
            transport_ns,
            "nitrate_diffusivity_water",
        )
        self._molecular_diffusivity_temperature_exponent = self._fetch_default(
            transport_ns,
            "molecular_diffusivity_temperature_exponent",
        )
        self._millington_quirk_exponent_numerator = self._fetch_default(
            "pedosphere.physical",
            "millington_quirk_exponent_numerator",
        )
        self._millington_quirk_exponent_denominator = self._fetch_default(
            "pedosphere.physical",
            "millington_quirk_exponent_denominator",
        )
        self._liter_to_cubic_meter = float(
            get_physical_constant(
                "relic.physical.constants.conversions.liter_to_cubic_meter"
            )
        )
        self._reference_temperature = float(get_physical_constant("reference_temperature"))
    
    def get_dependencies(self) -> Dict[str, List[str]]:
        """Get required input dependencies."""
        return self._merge_declared_dependencies({"required": [], "optional": []})
    
    def get_output_specifications(self) -> Dict[str, FluxSpecification]:
        """Get output variable specifications."""
        return {
            "nitrogen_leaching_flux": FluxSpecification(
                name="nitrogen_leaching_flux",
                units="g N m-2 s-1",
                direction="output",
                flux_type="leaching",
                description="NO3 mass flux through the free-drainage bottom boundary",
            ),
            "carbon_leaching_flux": FluxSpecification(
                name="carbon_leaching_flux",
                units="g C m-2 s-1",
                direction="output",
                flux_type="leaching",
                description="DOC mass flux through the free-drainage bottom boundary",
            ),
            "no3_concentration": FluxSpecification(
                name="no3_concentration",
                units="g N m-3",
                direction="internal",
                flux_type="concentration",
                description="NO3 concentration profile"
            ),
            "doc_concentration": FluxSpecification(
                name="doc_concentration", 
                units="g C m-3",
                direction="internal",
                flux_type="concentration",
                description="DOC concentration profile"
            ),
        }

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Return structural audit degrees of freedom per solute species."""

        value = int(n_dofs) if n_dofs is not None else int(self.n_layers)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for soil solute transport.

        References:
          - Bear (1972), Dynamics of Fluids in Porous Media.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          c_dot = -D_h(q_adv + q_diff) + s_h, q_diff = -star_D d_h c_h.
        Parameter meanings:
          n_layers defines the local support for NO3 and DOC concentration cochains.
        """

        n_layers = self._structural_n_dofs()
        n_species_dofs = 2 * n_layers
        return (
            CochainSpaceSpec(
                complex_id="pedosphere_solute_transport",
                degree=0,
                primality="primal",
                n_dofs=n_layers,
                units="g N m-3",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="no3_concentration",
                orientation_convention="downward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_solute_transport",
                degree=0,
                primality="primal",
                n_dofs=n_layers,
                units="g C m-3",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="doc_concentration",
                orientation_convention="downward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_solute_transport",
                degree=1,
                primality="primal",
                n_dofs=n_layers,
                units="m s-1",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="solute_advective_velocity",
                orientation_convention="downward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_solute_transport",
                degree=1,
                primality="primal",
                n_dofs=n_species_dofs,
                units="g m-4",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="solute_concentration_gradient",
                orientation_convention="downward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_solute_transport",
                degree=2,
                primality="dual",
                n_dofs=n_species_dofs,
                units="g m-2 s-1",
                extensivity="integral",
                physical_domain="pedosphere",
                semantic_name="solute_advection_diffusion_flux",
                orientation_convention="downward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "pedosphere.grid.n_layers",
            "pedosphere.grid.soil_depth",
            "pedosphere.transport.cfl_number",
            "pedosphere.transport.layer_thickness_ratio",
            "pedosphere.transport.time_step_maximum",
            "pedosphere.transport.default_molecular_diffusivity_water",
            "pedosphere.transport.molecular_diffusivity_temperature_exponent",
            "pedosphere.transport.nitrate_diffusivity_water",
            "pedosphere.transport.doc_diffusivity_water",
            "pedosphere.transport.nitrate_longitudinal_dispersivity",
            "pedosphere.transport.doc_longitudinal_dispersivity",
            "pedosphere.transport.nitrate_distribution_coefficient",
            "pedosphere.transport.doc_distribution_coefficient",
            "pedosphere.physical.millington_quirk_exponent_numerator",
            "pedosphere.physical.millington_quirk_exponent_denominator",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the solute dispersion Hodge contract.

        References:
          - Bear (1972), hydrodynamic dispersion.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_diff = -star_D d_h c_h with star_D = diag(D_i) and D_i >= 0.
        Parameter meanings:
          Default water molecular diffusivity initializes the passive structural audit.
        """

        n_layers = self._structural_n_dofs(n_dofs=n_dofs)
        value = torch.as_tensor(
            self.diffusivity,
            dtype=torch.float32,
            device=self.layer_thickness.device,
        ).reshape(1)
        values = value.expand(2 * n_layers)
        if bool(torch.any(values < 0.0)):
            raise ValueError("Solute diffusivity must be non-negative for Hodge assembly.")

        domain = CochainSpaceSpec(
            complex_id="pedosphere_solute_transport",
            degree=1,
            primality="primal",
            n_dofs=2 * n_layers,
            units="g m-4",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="solute_concentration_gradient",
            orientation_convention="downward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="pedosphere_solute_transport",
            degree=2,
            primality="dual",
            n_dofs=2 * n_layers,
            units="g m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="solute_advection_diffusion_flux",
            orientation_convention="downward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=("pedosphere.transport.default_molecular_diffusivity_water",),
            metric_keys=("pedosphere.grid.soil_depth", "pedosphere.grid.n_layers"),
            material_law="solute_hydrodynamic_dispersion",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build solute transport Hodge terms for Phase 8 IPHS assembly."""

        n_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        if n_dofs is not None:
            n_dofs = max(1, int(n_dofs) // 2)
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def _has_solute_state(self, state: Mapping[str, Any]) -> bool:
        """Return whether NO3 and DOC concentration cochains are available."""

        return (
            isinstance(state, Mapping)
            and isinstance(state.get("no3_concentration"), torch.Tensor)
            and isinstance(state.get("doc_concentration"), torch.Tensor)
        )

    def _has_transport_state(self, state: Mapping[str, Any]) -> bool:
        """Return whether state contains fields needed for dispersion assembly."""

        return self._has_solute_state(state) and all(
            key in state
            for key in (
                "soil_water_content",
                "darcy_flux",
                "bulk_density",
                "porosity",
                "soil_temperature",
            )
        )

    def _solute_concentration_vector(self, state: Mapping[str, Any]) -> torch.Tensor:
        """Return stacked NO3 and DOC concentration cochains."""

        no3_conc = self._extract_layer_values(state["no3_concentration"])
        doc_conc = self._extract_layer_values(state["doc_concentration"])
        return torch.cat([no3_conc.reshape(-1), doc_conc.reshape(-1)], dim=0)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return optional storage terms for structural IPHS audits."""

        if not self._has_solute_state(state):
            return {
                "storage_functional": "H_h = 0.5 * c_h.T c_h",
                "effort": "e_h = delta H_h / delta c_h = c_h",
            }
        concentration = self._solute_concentration_vector(state)
        return {
            "storage_functional": 0.5 * torch.sum(concentration * concentration),
            "effort": concentration,
        }

    def _dispersion_terms(self, state: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Assemble current solute concentrations and dispersion coefficients."""

        theta = self._extract_layer_values(state["soil_water_content"])
        q = self._extract_layer_values(state["darcy_flux"])
        porosity = self._extract_layer_values(state["porosity"])
        no3_conc = self._extract_layer_values(state["no3_concentration"])
        doc_conc = self._extract_layer_values(state["doc_concentration"])
        temperature = self._extract_layer_values(state["soil_temperature"])
        no3_dispersion = self._calculate_dispersion_coefficient(
            theta,
            q,
            temperature,
            porosity,
            "NO3",
        )
        doc_dispersion = self._calculate_dispersion_coefficient(
            theta,
            q,
            temperature,
            porosity,
            "DOC",
        )
        return {
            "no3_conc": no3_conc,
            "doc_conc": doc_conc,
            "no3_dispersion": no3_dispersion,
            "doc_dispersion": doc_dispersion,
        }

    def _interface_dissipation(
        self,
        concentration: torch.Tensor,
        dispersion: torch.Tensor,
    ) -> torch.Tensor:
        """Compute non-negative solute diffusion-dispersion dissipation.

        References:
          - Bear (1972), dispersion flux in porous media.
          - LeVeque (2002), finite-volume diffusion stability.
        Mathematical expression:
          sigma_h = sum_i D_i+1/2 * ((c_i+1 - c_i) / dz)^2.
        Parameter meanings:
          concentration: layer solute concentration [g m-3].
          dispersion: layer hydrodynamic dispersion [m2 s-1].
          dz: mean layer spacing [m].
        """

        if concentration.numel() < 2:
            return concentration.new_zeros(())
        centers = self.layer_centers.to(concentration.device, concentration.dtype)
        gradient = (concentration[1:] - concentration[:-1]) / (
            centers[1:] - centers[:-1]
        )
        interface_dispersion = 0.5 * (dispersion[:-1] + dispersion[1:])
        return torch.sum(interface_dispersion * gradient * gradient)

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return optional non-negative solute dispersion dissipation terms."""

        if not self._has_transport_state(state):
            return {
                "dissipation_rate": "sigma_h = X_h.T star_D X_h",
                "thermodynamic_force": "X_h = d_h c_h",
            }
        terms = self._dispersion_terms(state)
        return {
            "dissipation_rate": self._interface_dissipation(
                terms["no3_conc"], terms["no3_dispersion"]
            ) + self._interface_dissipation(
                terms["doc_conc"], terms["doc_dispersion"]
            ),
            "thermodynamic_force": "X_h = d_h c_h",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no assembled reversible block for this level 3 ADR view."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive solute dispersion mobility contribution.

        References:
          - Bear (1972), hydrodynamic dispersion in porous media.
          - van der Schaft and Maschke (1995), dissipative port-Hamiltonian form.
        Mathematical expression:
          M_h = diag(D_i), D_i >= 0, sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          D_i is assembled from ParameterManager-backed diffusivity,
          dispersivity, temperature exponent, theta, and Darcy flux.
        """

        if self._has_transport_state(state):
            terms = self._dispersion_terms(state)
            values = torch.cat(
                [
                    terms["no3_dispersion"].reshape(-1),
                    terms["doc_dispersion"].reshape(-1),
                ],
                dim=0,
            ).to(dtype=torch.float32, device=self.layer_thickness.device)
        else:
            values = torch.diag(self.build_hodge_contract(strict=True).matrix)
        if bool(torch.any(values < 0.0)):
            raise ValueError("Solute dispersion mobility must be non-negative in M_h.")
        matrix = torch.diag(values)
        n_layers = int(values.numel() // 2)
        return {
            "state_vars": tuple(
                [f"no3_concentration[{idx}]" for idx in range(n_layers)]
                + [f"doc_concentration[{idx}]" for idx in range(n_layers)]
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare surface input and bottom leaching solute ports.

        References:
          - Bear (1972), boundary fluxes in porous media transport.
          - van der Schaft and Maschke (1995), port power pairing.
        Mathematical expression:
          P_port = e_h * f_h with e_h = c_boundary and f_h = q_boundary.
        Parameter meanings:
          Surface inputs and bottom leaching are existing RELIC matter ports.
        """

        effort_space = CochainSpaceSpec(
            complex_id="pedosphere_solute_transport",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="g m-3",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="boundary_solute_concentration",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="pedosphere_solute_transport",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="g m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="boundary_solute_flux",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="solute_surface_inflow",
                effort_var="surface_solute_concentration",
                flow_var="solute_surface_input_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="g2 m-5 s-1",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="solute_bottom_leaching",
                effort_var="bottom_solute_concentration",
                flow_var="bottom_leaching_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="g2 m-5 s-1",
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
    
    def validate_inputs(self, state: StateDict) -> List[str]:
        """Validate the closed canonical input surface without fallbacks."""

        errors: List[str] = []
        layer_keys = {
            "soil_water_content",
            "darcy_flux",
            "no3_concentration",
            "doc_concentration",
            "bulk_density",
            "porosity",
            "soil_temperature",
        }
        scalar_keys = {
            "surface_no3_concentration",
            "surface_doc_concentration",
        }
        for key in self.get_required_inputs():
            if key not in state:
                errors.append(f"Missing required input: {key}")
                continue
            value = state[key]
            if not isinstance(value, torch.Tensor):
                errors.append(f"{key} must be torch.Tensor")
                continue
            if key in layer_keys and value.numel() != self.n_layers:
                errors.append(
                    f"{key} must contain exactly {self.n_layers} layer values"
                )
                continue
            if key in scalar_keys and value.numel() != 1:
                errors.append(f"{key} must be a scalar tensor")
                continue
            if not bool(torch.all(torch.isfinite(value))):
                errors.append(f"{key} must be finite")

        if errors:
            return errors

        theta = state["soil_water_content"].reshape(-1)
        porosity = state["porosity"].reshape(-1)
        if bool(torch.any(theta <= 0.0)) or bool(torch.any(theta > 1.0)):
            errors.append("soil_water_content must be in (0, 1]")
        if bool(torch.any(porosity <= 0.0)) or bool(torch.any(porosity > 1.0)):
            errors.append("porosity must be in (0, 1]")
        if bool(torch.any(theta > porosity)):
            errors.append("soil_water_content must not exceed porosity")
        if bool(torch.any(state["darcy_flux"] < 0.0)):
            errors.append("darcy_flux must be non-negative for bottom leaching")
        for key in (
            "no3_concentration",
            "doc_concentration",
            "surface_no3_concentration",
            "surface_doc_concentration",
        ):
            if bool(torch.any(state[key] < 0.0)):
                errors.append(f"{key} must be non-negative")
        if bool(torch.any(state["bulk_density"] <= 0.0)):
            errors.append("bulk_density must be positive")
        if bool(torch.any(state["bulk_density"] > 3000.0)):
            errors.append("bulk_density must not exceed 3000 kg m-3")
        if bool(torch.any(state["soil_temperature"] <= 0.0)):
            errors.append("soil_temperature must be positive")
        return errors

    def _extract_layer_values(
        self,
        tensor: torch.Tensor,
        variable_name: str = "layer variable",
    ) -> torch.Tensor:
        """Return an exact layer vector without broadcast or truncation."""

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{variable_name} must be torch.Tensor")
        if tensor.numel() != self.n_layers:
            raise ValueError(
                f"{variable_name} must contain exactly {self.n_layers} layer values"
            )
        return tensor.reshape(self.n_layers)

    def _resolve_transport_inputs(
        self,
        state: StateDict,
        dt: torch.Tensor | float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        """Resolve validated canonical tensors on one device and dtype."""

        errors = self.validate_inputs(state)
        if errors:
            raise ValueError("Input validation failed: " + "; ".join(errors))
        theta = self._extract_layer_values(
            state["soil_water_content"],
            "soil_water_content",
        )
        device = theta.device
        dtype = theta.dtype

        def _layer(key: str) -> torch.Tensor:
            value = self._extract_layer_values(state[key], key)
            if value.device != device:
                raise ValueError(f"{key} must use the soil_water_content device")
            return value.to(dtype=dtype)

        def _scalar(key: str) -> torch.Tensor:
            value = state[key]
            if value.device != device:
                raise ValueError(f"{key} must use the soil_water_content device")
            return value.to(dtype=dtype).reshape(())

        dt_tensor = torch.as_tensor(dt, dtype=dtype, device=device)
        if dt_tensor.numel() != 1 or not bool(torch.all(torch.isfinite(dt_tensor))):
            raise ValueError("dt must be one finite scalar")
        dt_value = float(dt_tensor.detach().reshape(()))
        if dt_value <= 0.0:
            raise ValueError("dt must be positive")
        return (
            theta,
            _layer("darcy_flux"),
            _layer("no3_concentration"),
            _layer("doc_concentration"),
            _layer("bulk_density"),
            _layer("porosity"),
            _layer("soil_temperature"),
            _scalar("surface_no3_concentration"),
            _scalar("surface_doc_concentration"),
            dt_value,
        )

    def _storage_capacity(
        self,
        theta: torch.Tensor,
        bulk_density: torch.Tensor,
        species_name: str,
    ) -> torch.Tensor:
        """Return B = theta + rho_b Kd after the required SI conversion."""

        if species_name == "NO3":
            kd_l_per_kg = self._nitrate_distribution_coefficient
        elif species_name == "DOC":
            kd_l_per_kg = self._doc_distribution_coefficient
        else:
            raise ValueError(f"Unsupported canonical solute species: {species_name}")
        if not self.enable_sorption:
            kd_l_per_kg = 0.0
        kd_m3_per_kg = torch.as_tensor(
            kd_l_per_kg * self._liter_to_cubic_meter,
            dtype=theta.dtype,
            device=theta.device,
        )
        return theta + bulk_density * kd_m3_per_kg

    def _calculate_dispersion_coefficient(
        self,
        theta: torch.Tensor,
        q: torch.Tensor,
        temperature: torch.Tensor,
        porosity: torch.Tensor,
        species_name: str,
    ) -> torch.Tensor:
        """Return the bulk dispersion coefficient K [m2 s-1].

        References: Millington and Quirk (1961); HYDRUS-1D 4.16 manual.
        Mathematical expression: K = alpha_L abs(q) + D_w(T) theta^(10/3)/phi^2.
        Parameter meanings: alpha_L [m], q [m s-1], D_w [m2 s-1], theta and
        phi [m3 m-3], T and T_ref [K], and n [-] in D_w=D_ref(T/T_ref)^n.
        """

        if species_name == "NO3":
            molecular_diffusivity = self._nitrate_diffusivity_water
            dispersivity = self._nitrate_dispersivity
        elif species_name == "DOC":
            molecular_diffusivity = self._doc_diffusivity_water
            dispersivity = self._doc_dispersivity
        else:
            raise ValueError(f"Unsupported canonical solute species: {species_name}")
        reference_temperature = torch.as_tensor(
            self._reference_temperature,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        exponent = torch.as_tensor(
            self._molecular_diffusivity_temperature_exponent,
            dtype=temperature.dtype,
            device=temperature.device,
        )
        d_water = torch.as_tensor(
            molecular_diffusivity,
            dtype=temperature.dtype,
            device=temperature.device,
        ) * torch.pow(temperature / reference_temperature, exponent)
        numerator = torch.as_tensor(
            self._millington_quirk_exponent_numerator,
            dtype=theta.dtype,
            device=theta.device,
        )
        denominator = torch.as_tensor(
            self._millington_quirk_exponent_denominator,
            dtype=theta.dtype,
            device=theta.device,
        )
        molecular_bulk = d_water * torch.pow(theta, numerator) / torch.pow(
            porosity,
            denominator,
        )
        mechanical_bulk = torch.as_tensor(
            dispersivity,
            dtype=q.dtype,
            device=q.device,
        ) * torch.abs(q)
        return molecular_bulk + mechanical_bulk

    @staticmethod
    def _harmonic_interface_values(values: torch.Tensor) -> torch.Tensor:
        """Return harmonic means at internal cell faces."""

        if values.numel() < 2:
            return values.new_empty((0,))
        denominator = values[:-1] + values[1:]
        safe_denominator = torch.where(
            denominator > 0.0,
            denominator,
            torch.ones_like(denominator),
        )
        return torch.where(
            denominator > 0.0,
            2.0 * values[:-1] * values[1:] / safe_denominator,
            torch.zeros_like(denominator),
        )

    def _face_fluxes(
        self,
        concentration: torch.Tensor,
        q: torch.Tensor,
        bulk_dispersion: torch.Tensor,
        surface_concentration: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble downward-positive finite-volume face fluxes.

        References: Bear (1972); HYDRUS-1D 4.16 manual. Mathematical
        expression: J=q C_upwind-K dC/dz, J_top=q_top C_surface, and
        J_bottom=q_bottom C_bottom. Parameters are C [g element m-3], q
        [m s-1], K [m2 s-1], and z [m].
        """

        faces = concentration.new_empty((self.n_layers + 1,))
        faces[0] = q[0] * surface_concentration
        if self.n_layers > 1:
            q_face = 0.5 * (q[:-1] + q[1:])
            centers = self.layer_centers.to(
                dtype=concentration.dtype,
                device=concentration.device,
            )
            distance = centers[1:] - centers[:-1]
            dispersion_face = self._harmonic_interface_values(bulk_dispersion)
            gradient = (concentration[1:] - concentration[:-1]) / distance
            faces[1:-1] = q_face * concentration[:-1] - dispersion_face * gradient
        faces[-1] = q[-1] * concentration[-1]
        return faces

    def _calculate_stable_timestep(
        self,
        storage_capacity: torch.Tensor,
        q: torch.Tensor,
        bulk_dispersion: torch.Tensor,
        requested_dt: float,
    ) -> float:
        """Return a capacity-weighted explicit finite-volume stability step."""

        dz = self.layer_thickness.to(
            dtype=storage_capacity.dtype,
            device=storage_capacity.device,
        )
        out_coefficient = torch.zeros_like(storage_capacity)
        if self.n_layers > 1:
            q_face = 0.5 * (q[:-1] + q[1:])
            out_coefficient[:-1] = out_coefficient[:-1] + q_face
            centers = self.layer_centers.to(
                dtype=storage_capacity.dtype,
                device=storage_capacity.device,
            )
            diffusive_coefficient = self._harmonic_interface_values(
                bulk_dispersion
            ) / (centers[1:] - centers[:-1])
            out_coefficient[:-1] = out_coefficient[:-1] + diffusive_coefficient
            out_coefficient[1:] = out_coefficient[1:] + diffusive_coefficient
        out_coefficient[-1] = out_coefficient[-1] + q[-1]
        active = out_coefficient > 0.0
        if bool(torch.any(active)):
            candidates = (
                self.cfl_number
                * storage_capacity[active]
                * dz[active]
                / out_coefficient[active]
            )
            stable = float(torch.min(candidates).detach())
        else:
            stable = self.max_timestep
        stable = min(stable, self.max_timestep, requested_dt)
        if stable <= 0.0:
            raise ValueError("Finite-volume stable timestep must be positive")
        return stable

    def _advance_species(
        self,
        concentration: torch.Tensor,
        storage_capacity: torch.Tensor,
        q: torch.Tensor,
        bulk_dispersion: torch.Tensor,
        surface_concentration: torch.Tensor,
        dt_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one species and integrate both open matter ports."""

        stable_dt = self._calculate_stable_timestep(
            storage_capacity,
            q,
            bulk_dispersion,
            dt_value,
        )
        n_substeps = max(
            1,
            int(torch.ceil(torch.as_tensor(dt_value / stable_dt)).detach()),
        )
        actual_dt = dt_value / n_substeps
        dz = self.layer_thickness.to(
            dtype=concentration.dtype,
            device=concentration.device,
        )
        current = concentration
        surface_mass = concentration.new_zeros(())
        bottom_mass = concentration.new_zeros(())
        for _ in range(n_substeps):
            faces = self._face_fluxes(
                current,
                q,
                bulk_dispersion,
                surface_concentration,
            )
            current = (
                storage_capacity * current
                - actual_dt * (faces[1:] - faces[:-1]) / dz
            ) / storage_capacity
            if not bool(torch.all(torch.isfinite(current))):
                raise ValueError("Solute concentration became non-finite")
            if bool(torch.any(current < 0.0)):
                raise ValueError(
                    "Solute concentration became negative; reduce cfl_number"
                )
            surface_mass = surface_mass + faces[0] * actual_dt
            bottom_mass = bottom_mass + faces[-1] * actual_dt
        return current, surface_mass, bottom_mass

    @staticmethod
    def _update(
        value: torch.Tensor,
        *,
        per_timestep: bool,
        unit: str,
        source_functor: str,
    ) -> UpdateItem:
        """Create a non-global-ledger REPLACE update with runtime semantics.

        The local solute ledger uses B * C * dz geometry. StateManager's
        global elemental ledger is areal and cannot infer that geometry from a
        concentration UpdateItem, so these outputs intentionally carry no
        element tag.
        """

        return UpdateItem(
            value=value,
            mode=UpdateMode.REPLACE,
            basis="per_step" if per_timestep else "instantaneous",
            per_timestep=per_timestep,
            unit=unit,
            source_functor=source_functor,
        )

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: torch.Tensor | float,
        **kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        """Advance NO3 and DOC with one exact boundary-ledger formula.

        References: Simunek et al. (2008); HYDRUS-1D 4.16 manual; Bear (1972).
        Mathematical expression: B_i C_i^(n+1)=B_i C_i^n-
        dt(J_i+1/2-J_i-1/2)/dz_i. Parameter meanings: B [-], C [g element
        m-3], J [g element m-2 s-1], dz [m], and dt [s]. Top Cauchy input
        and bottom free-drainage export are the only open-system routes.
        """

        # References: Simunek et al. (2008); Bear (1972).
        # Mathematical expression: B_i dC_i/dt = -(J_i+1/2-J_i-1/2)/dz_i.
        # Parameter meanings: B [1], C [g element m-3], J [g element m-2 s-1].
        _ = time, kwargs
        (
            theta,
            q,
            no3_initial,
            doc_initial,
            bulk_density,
            porosity,
            temperature,
            surface_no3,
            surface_doc,
            dt_value,
        ) = self._resolve_transport_inputs(state, dt)
        dz = self.layer_thickness.to(dtype=theta.dtype, device=theta.device)
        no3_capacity = self._storage_capacity(theta, bulk_density, "NO3")
        doc_capacity = self._storage_capacity(theta, bulk_density, "DOC")
        no3_dispersion = self._calculate_dispersion_coefficient(
            theta, q, temperature, porosity, "NO3"
        )
        doc_dispersion = self._calculate_dispersion_coefficient(
            theta, q, temperature, porosity, "DOC"
        )
        no3_final, no3_surface_mass, no3_bottom_mass = self._advance_species(
            no3_initial,
            no3_capacity,
            q,
            no3_dispersion,
            surface_no3,
            dt_value,
        )
        doc_final, doc_surface_mass, doc_bottom_mass = self._advance_species(
            doc_initial,
            doc_capacity,
            q,
            doc_dispersion,
            surface_doc,
            dt_value,
        )
        no3_residual = (
            torch.sum(no3_capacity * no3_final * dz)
            - torch.sum(no3_capacity * no3_initial * dz)
            - no3_surface_mass
            + no3_bottom_mass
        )
        doc_residual = (
            torch.sum(doc_capacity * doc_final * dz)
            - torch.sum(doc_capacity * doc_initial * dz)
            - doc_surface_mass
            + doc_bottom_mass
        )
        dt_tensor = torch.as_tensor(dt_value, dtype=theta.dtype, device=theta.device)
        return {
            "nitrogen_leaching_flux": self._update(
                no3_bottom_mass / dt_tensor,
                per_timestep=False,
                unit="g N m-2 s-1",
                source_functor=self.name,
            ),
            "carbon_leaching_flux": self._update(
                doc_bottom_mass / dt_tensor,
                per_timestep=False,
                unit="g C m-2 s-1",
                source_functor=self.name,
            ),
            "no3_concentration": self._update(
                no3_final,
                per_timestep=True,
                unit="g N m-3",
                source_functor=self.name,
            ),
            "doc_concentration": self._update(
                doc_final,
                per_timestep=True,
                unit="g C m-3",
                source_functor=self.name,
            ),
            "diagnostics.solute_transport.nitrogen_surface_input_flux": self._update(
                no3_surface_mass / dt_tensor,
                per_timestep=False,
                unit="g N m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.solute_transport.carbon_surface_input_flux": self._update(
                doc_surface_mass / dt_tensor,
                per_timestep=False,
                unit="g C m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.solute_transport.nitrogen_mass_balance_residual": self._update(
                no3_residual,
                per_timestep=True,
                unit="g N m-2",
                source_functor=self.name,
            ),
            "diagnostics.solute_transport.carbon_mass_balance_residual": self._update(
                doc_residual,
                per_timestep=True,
                unit="g C m-2",
                source_functor=self.name,
            ),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: Mapping[str, Any],
        dt: Optional[torch.Tensor | float] = None,
    ) -> Dict[str, float]:
        """Return the maximum integrated NO3/DOC ledger residual."""

        _ = input_state, dt
        keys = (
            "diagnostics.solute_transport.nitrogen_mass_balance_residual",
            "diagnostics.solute_transport.carbon_mass_balance_residual",
        )
        if not all(key in output_updates for key in keys):
            return {}
        magnitudes = []
        for key in keys:
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            tensor = torch.as_tensor(value)
            if not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"{key} must be finite")
            magnitudes.append(float(torch.max(torch.abs(tensor)).detach()))
        return {"mass": max(magnitudes)}


__all__ = ["SoluteAdvectionDiffusion1D"]
