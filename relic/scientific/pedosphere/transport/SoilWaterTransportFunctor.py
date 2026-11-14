"""Soil Water Transport Functor

This module implements one-dimensional Richards soil-water transport in the
pedosphere with the van Genuchten--Mualem hydraulic closure.

Mathematical Expression:
    Richards equation: dtheta/dt = div(K(theta) * grad(h + z)) + S
    where K(theta) is unsaturated hydraulic conductivity function
    h is pressure head, z is elevation

Parameters:
    config : Optional[dict]
        Parameter overrides; the one-dimensional van Genuchten-Mualem column is the
        only production law (no dimension or closure selector keyword).
    device : Optional[torch.device]
        Computation device
"""

from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.computation.solvers import SolverFactory
from relic.core.abstractions import FluxDict, StateDict
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
from relic.scientific.hydrosphere.solvers import RichardsNumericalSolver
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin
from relic.utils.soil_hydraulics import (
    effective_saturation,
    van_genuchten_mualem_conductivity,
    van_genuchten_pressure_head,
    van_genuchten_water_retention,
)
from relic.utils.physical_constants import get_physical_constant

class SoilWaterTransportFunctor(PedosphereContractMixin, TransportMechanismFunctor):
    """Soil water transport functor for Richards equation and boundary fluxes."""

    TRANSPORT_TYPE = "diffusion"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    # One-dimensional vertical column: the only production law (van Genuchten-Mualem).
    SPATIAL_DIMS = 1

    def __init__(
        self,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize soil water transport functor."""
        if config is None:
            config = {}
        
        # Set up metadata
        metadata = FunctorMetadata(
            name="soil_water_transport",
            version="1.0.0",
            description="One-dimensional Richards soil-water transport with van Genuchten--Mualem hydraulics",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "theta_dot = -D_h q_h + G_h u_h; q_h = -star_K d(h + z)"
            ),
            references=[
                "Darcy, H. (1856). Les fontaines publiques de la ville de Dijon.",
                "Richards, L.A. (1931). Capillary conduction of liquids through porous mediums.",
                "Celia, M.A. et al. (1990). A general mass-conservative numerical solution for the unsaturated flow equation.",
                "van Genuchten, M.Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils.",
                "Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media.",
                "Hirani, A.N. (2003). Discrete Exterior Calculus."
            ],
            parameters={
                "pedosphere.grid.dz": "Vertical grid spacing [m]",
                "pedosphere.hydraulic.saturated_conductivity": (
                    "Saturated hydraulic conductivity [m s-1]"
                ),
                "pedosphere.moisture.wilting_point": "Wilting point water content [m3 m-3]",
                "pedosphere.soil.porosity": "Soil porosity [m3 m-3]",
                "pedosphere.van_genuchten.alpha_vg": "Van Genuchten alpha parameter [m-1]",
                "pedosphere.van_genuchten.n_vg": "Van Genuchten n parameter [dimensionless]",
                "pedosphere.transport.gravity_head_gradient": (
                    "Gravity head gradient [dimensionless]"
                ),
                "transport.numerical.min_timestep": "Minimum stable time step [s]",
                "transport.numerical.richards_tolerance": (
                    "Richards Picard convergence tolerance [dimensionless]"
                ),
                "transport.numerical.max_richards_iterations": (
                    "Maximum Richards Picard iterations [dimensionless]"
                ),
                "pedosphere.numerical.cfl_number": "CFL safety coefficient [dimensionless]",
                "pedosphere.numerical.epsilon_length": "Finite difference perturbation length [m]",
                "pedosphere.numerical.epsilon_diffusivity": "Diffusivity epsilon [dimensionless]",
                "pedosphere.numerical.water_capacity_epsilon": (
                    "Minimum water capacity scale [dimensionless]"
                ),
                "pedosphere.numerical.harmonic_mean_factor": (
                    "Harmonic interface mean coefficient [dimensionless]"
                ),
                "pedosphere.numerical.linear_solver_tolerance": (
                    "Tridiagonal solver tolerance [dimensionless]"
                ),
                "pedosphere.numerical.under_relaxation_factor": (
                    "Picard under-relaxation factor [dimensionless]"
                ),
            },
            input_variables=[
                "soil_moisture",
                "pressure_head",
                "infiltration_flux",
                "evapotranspiration_flux",
            ],
            output_variables=[
                "soil_moisture",
                "diagnostics.soil_water_transport.clipping_residual",
                "diagnostics.soil_water_transport.water_balance_residual",
                "diagnostics.soil_water_transport.darcy_flux",
                "diagnostics.soil_water_transport.richards_iterations",
            ],
            conservation_laws=["mass", "water"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            spatial_dimensions=self.SPATIAL_DIMS,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["pedosphere.hydrology", "pedosphere.biology"]
        )
        
        super().__init__(name="soil_water_transport", config=config, metadata=metadata)

        self.optional_input_variables = {"pressure_head"}
        
        # Set device
        self.device = device if device is not None else (torch.device("cpu"))
        
        # Set up the one-dimensional grid from the ParameterManager SSOT.
        dz = self.get_parameter_value(
            'dz',
            namespace='pedosphere.grid',
            required=True,
        )
        self.grid_spacing = (dz,)
        
        # Soil hydraulic parameters
        self.saturated_conductivity = self.get_parameter_value(
            'saturated_conductivity',
            namespace='pedosphere.hydraulic',
            required=True,
        )
        
        self.wilting_point = self.get_parameter_value(
            'wilting_point',
            namespace='pedosphere.moisture',
            required=True,
        )
        
        self.porosity = self.get_parameter_value(
            'porosity',
            namespace='pedosphere.soil',
            required=True,
        )

        self.water_density = float(get_physical_constant("water_density"))
        if not torch.isfinite(torch.as_tensor(self.water_density)) or self.water_density <= 0.0:
            raise ValueError("Physical constant water_density must be positive and finite.")
        
        # van Genuchten parameters
        alpha_vg = self.get_parameter_value(
            'alpha_vg',
            namespace='pedosphere.van_genuchten',
            expected_type=float,
            required=True,
        )
        n_vg = self.get_parameter_value(
            'n_vg',
            namespace='pedosphere.van_genuchten',
            expected_type=float,
            required=True,
        )
        if isinstance(alpha_vg, Tensor):
            if alpha_vg.numel() != 1:
                raise ValueError("Expected scalar for pedosphere.van_genuchten.alpha_vg")
            alpha_vg = float(alpha_vg.detach())
        if isinstance(n_vg, Tensor):
            if n_vg.numel() != 1:
                raise ValueError("Expected scalar for pedosphere.van_genuchten.n_vg")
            n_vg = float(n_vg.detach())
        self.alpha_vg = float(alpha_vg)
        self.n_vg = float(n_vg)
        
        # Numerical parameters
        self.min_timestep = self.get_parameter_value(
            'min_timestep',
            namespace='transport.numerical',
            required=True,
        )
        self.richards_tolerance = self.get_parameter_value(
            'richards_tolerance',
            namespace='transport.numerical',
            required=True,
        )
        self.max_richards_iterations = self.get_parameter_value(
            'max_richards_iterations',
            namespace='transport.numerical',
            required=True,
        )
        self.gravity_head_gradient = self.get_parameter_value(
            'gravity_head_gradient',
            namespace='pedosphere.transport',
            units='dimensionless',
            required=True,
        )
        self.cfl_number = self.get_parameter_value(
            'cfl_number',
            namespace='pedosphere.numerical',
            units='dimensionless',
            required=True,
        )
        self.epsilon_length = self.get_parameter_value(
            'epsilon_length',
            namespace='pedosphere.numerical',
            units='m',
            required=True,
        )
        self.epsilon_diffusivity = self.get_parameter_value(
            'epsilon_diffusivity',
            namespace='pedosphere.numerical',
            units='dimensionless',
            required=True,
        )
        self.saturation_epsilon = self.epsilon_diffusivity
        self.water_capacity_epsilon = self.get_parameter_value(
            'water_capacity_epsilon',
            namespace='pedosphere.numerical',
            units='dimensionless',
            required=True,
        )
        self.harmonic_mean_factor = self.get_parameter_value(
            'harmonic_mean_factor',
            namespace='pedosphere.numerical',
            units='dimensionless',
            required=True,
        )
        self.linear_solver_tolerance = self.get_parameter_value(
            'linear_solver_tolerance',
            namespace='pedosphere.numerical',
            units='dimensionless',
            required=True,
        )
        self.under_relaxation_factor = self.get_parameter_value(
            'under_relaxation_factor',
            namespace='pedosphere.numerical',
            units='dimensionless',
            required=True,
        )
        self._top_boundary = None
        self._bottom_boundary = None

        solver_tolerance = self.linear_solver_tolerance
        if isinstance(solver_tolerance, torch.Tensor):
            solver_tolerance = float(solver_tolerance.detach())

        # References: Thomas (1949) tridiagonal solver; used in Richards Picard steps.
        # Mathematical expression: A x = d for tri-diagonal system from 1D Richards discretization.
        # Parameter meanings: rtol, atol are solver tolerances [dimensionless].
        self.thomas_solver = SolverFactory.create(
            solver_name="thomas_algorithm",
            name="soil_water_transport_tridiag_solver",
            config={
                "check_stability": True,
                "rtol": solver_tolerance,
                "atol": solver_tolerance,
            },
        )
        self._richards_solver = RichardsNumericalSolver(self)

    def get_dependencies(self) -> dict:
        """Return lagged boundary-flux dependencies without algebraic cycles.

        Surface runoff reads the committed soil-moisture state, while selected
        ET closures also depend on hydrologic state. Their previously committed
        boundary fluxes are therefore consumed when an in-model writer exists;
        external StateManager adapters may provide the same canonical ports.
        """

        return self._merge_declared_dependencies({
            "required": [],
            "optional": [],
            "lagged": [
                "surface_runoff",
                "penman_monteith_et",
                "priestley_taylor_et",
            ],
        })

    def get_dec_form_metadata(self, n_dofs: int) -> dict[str, dict[str, Any]]:
        """Return optional DEC form metadata for soil water transport.

        References:
          - Hirani (2003), Discrete Exterior Calculus.
          - Hirani, Nakshatrala, and Chaudhry (2008), DEC Darcy flow.
        Mathematical expression:
          Theta is a cellwise volumetric-water-content profile, and column
          storage is derived as W = sum_i(theta_i dz_i).
          Darcy flux is represented as a face cochain q = -star_K d(h + z).
        Parameter meanings:
          n_dofs: number of local structural degrees of freedom declared for audit.
        """

        n_structural_dofs = int(n_dofs)
        return {
            "soil_moisture": {
                "form_degree": 3,
                "primality": "primal",
                "extensivity": "intensive",
                "physical_domain": "pedosphere",
                "semantic_name": "soil_moisture_profile",
                "units": "m3 m-3",
                "n_dofs": n_structural_dofs,
                "orientation_convention": "outward_positive",
                "boundary_role": "interior",
            },
            "water_flux": {
                "form_degree": 2,
                "primality": "dual",
                "extensivity": "integral",
                "physical_domain": "pedosphere",
                "semantic_name": "darcy_water_flux",
                "units": "m s-1",
                "n_dofs": n_structural_dofs,
                "orientation_convention": "outward_positive",
                "boundary_role": "boundary",
            },
        }

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve structural degrees of freedom for DEC audit views."""

        if n_dofs is not None:
            value = int(n_dofs)
        elif hasattr(complex_bundle, "n_dofs"):
            value = int(getattr(complex_bundle, "n_dofs"))
        elif hasattr(self, "soil_layers"):
            value = int(self.soil_layers)
        else:
            conductivity = torch.as_tensor(
                self.saturated_conductivity
            ).reshape(-1)
            value = int(conductivity.numel())
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare the canonical DEC cochain spaces for soil water transport.

        References:
          - Hirani (2003), Discrete Exterior Calculus.
          - Hirani, Nakshatrala, and Chaudhry (2008), DEC Darcy flow.
        Mathematical expression:
          theta_h = {theta_i}, X_h = d(h + z), q_h = -star_K X_h,
          theta_dot = -D_h q_h + G_h u_h.
        Parameter meanings:
          n_dofs is inferred from soil_layers or saturated_conductivity support.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="pedosphere_soil_water",
                degree=3,
                primality="primal",
                n_dofs=n_dofs,
                units="m3 m-3",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="soil_moisture_profile",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_soil_water",
                degree=1,
                primality="primal",
                n_dofs=n_dofs,
                units="m",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="hydraulic_head_gradient",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_soil_water",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="m s-1",
                extensivity="integral",
                physical_domain="pedosphere",
                semantic_name="darcy_water_flux",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys required by the IPHS structural view: the
        metadata parameter keys in their declared order."""

        return tuple(self.metadata.parameters)

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
        conductivity: Optional[Any] = None,
    ) -> HodgeStarSpec:
        """Build the optional Darcy-Hodge structural contract.

        References:
          - Darcy (1856), groundwater flow law.
          - Hirani, Nakshatrala, and Chaudhry (2008), DEC Darcy flow.
        Mathematical expression:
          q_h = -star_K d(h + z), with star_K = diag(K_s) for this local audit
          contract. Passive transport requires star_K = star_K^T >= 0.
        Parameter meanings:
          conductivity: K_s [m s-1], loaded from ParameterManager or supplied
            by the existing compiler resolved-parameter mapping.
          n_dofs: number of cochain degrees of freedom in the audit view.
          strict: validate symmetry and positive semidefiniteness immediately.
        """

        device = getattr(self, "device", torch.device("cpu"))
        conductivity_value = (
            self.saturated_conductivity
            if conductivity is None
            else conductivity
        )
        conductivity = torch.as_tensor(
            conductivity_value,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        target_dofs = int(n_dofs) if n_dofs is not None else int(conductivity.numel())
        if target_dofs < 1:
            raise ValueError("n_dofs must be positive for soil water Hodge contract.")
        if conductivity.numel() == 1 and target_dofs > 1:
            conductivity = conductivity.expand(target_dofs)
        if conductivity.numel() != target_dofs:
            raise ValueError(
                "saturated_conductivity size must match n_dofs for soil water Hodge contract."
            )

        domain = CochainSpaceSpec(
            complex_id="pedosphere_soil_water",
            degree=1,
            primality="primal",
            n_dofs=target_dofs,
            units="m",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="hydraulic_head_gradient",
            orientation_convention="outward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="pedosphere_soil_water",
            degree=2,
            primality="dual",
            n_dofs=target_dofs,
            units="m s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="darcy_water_flux",
            orientation_convention="outward_positive",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(conductivity),
            parameter_keys=("pedosphere.hydraulic.saturated_conductivity",),
            metric_keys=("pedosphere.grid.dz",),
            material_law="darcy",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build the Darcy conductivity Hodge term for IPHS assembly.

        References:
          - Darcy (1856), groundwater flow law.
          - Hirani, Nakshatrala, and Chaudhry (2008), DEC Darcy flow.
        Mathematical expression:
          q_h = -star_K d(h + z), star_K = star_K^T >= 0.
        Parameter meanings:
          pedosphere.hydraulic.saturated_conductivity supplies star_K entries.
          pedosphere.grid.dz supplies the metric support for the local audit.
        """

        n_dofs = self._structural_n_dofs(complex_bundle=complex_bundle)
        key = "pedosphere.hydraulic.saturated_conductivity"
        if resolved_parameters and key in resolved_parameters:
            return (
                self.build_hodge_contract(
                    n_dofs=n_dofs,
                    strict=True,
                    conductivity=resolved_parameters[key],
                ),
            )
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return optional energy terms for structural IPHS audits."""

        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return optional entropy terms for structural IPHS audits."""

        return {}

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for Darcy-only transport."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive Darcy mobility contribution.

        References:
          - Darcy (1856), groundwater flow law.
          - van Genuchten (1980) and Mualem (1976), unsaturated conductivity.
          - van der Schaft and Maschke (1995), dissipative port-Hamiltonian form.
        Mathematical expression:
          M_h = diag(K(theta_i)), K(theta_i) >= 0, sigma = X_h^T M_h X_h >= 0.
        Parameter meanings:
          theta: soil moisture [m3 m-3]; saturated_conductivity: K_s [m s-1].
        """

        theta = state.get("soil_moisture") if isinstance(state, Mapping) else None
        if isinstance(theta, Tensor):
            conductivity = self._compute_hydraulic_conductivity(theta.to(self.device))
        else:
            conductivity = torch.as_tensor(
                self.saturated_conductivity,
                dtype=torch.float32,
                device=self.device,
            )
        conductivity = conductivity.reshape(-1)
        if bool(torch.any(conductivity < 0.0)):
            raise ValueError("Hydraulic conductivity must be non-negative in M_h.")
        matrix = torch.diag(conductivity)
        return {
            "state_vars": tuple(
                f"soil_moisture[{idx}]" for idx in range(int(conductivity.numel()))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare surface and drainage water flux ports.

        References:
          - Richards (1931), boundary flux form for unsaturated flow.
          - van der Schaft and Maschke (1995), port variables and power pairing.
        Mathematical expression:
          P_port = e_h * f_h with e_h = hydraulic head [m] and f_h = q_b [m s-1].
        Parameter meanings:
          infiltration_flux and evapotranspiration_flux are converted from
          kg m-2 s-1 to the volumetric surface port with water_density;
          free drainage or prescribed bottom flux drives the drainage port.
        """

        effort_space = CochainSpaceSpec(
            complex_id="pedosphere_soil_water",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="boundary_hydraulic_head",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="pedosphere_soil_water",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="m s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="boundary_water_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="surface_water_flux_boundary",
                effort_var="surface_hydraulic_head",
                flow_var="surface_water_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="m2 s-1",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="bottom_drainage_boundary",
                effort_var="bottom_hydraulic_head",
                flow_var="bottom_water_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="m2 s-1",
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

    def _get_required_boundary_flux(
        self,
        state: StateDict,
        key: str,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """Return one required non-negative boundary mass flux.

        References:
          - Richards (1931), prescribed-flux surface boundary.
          - Allen et al. (1998), evapotranspiration as an outward water flux.
        Mathematical expression:
          q_v = F_w / rho_w.
        Parameter meanings:
          F_w: surface water mass flux [kg m-2 s-1].
          rho_w: liquid-water density [kg m-3] from the physical constants SSOT.
          q_v: volumetric boundary flux [m s-1].
        Open-system route:
          Infiltration enters and evapotranspiration leaves through the surface
          water matter port.
        """

        value = state.get(key)
        if value is None:
            raise ValueError(
                f"State must contain required boundary flux '{key}'."
            )
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must be a torch.Tensor scalar.")
        value = value.to(device=self.device, dtype=template.dtype)
        if value.numel() != 1:
            raise ValueError(f"{key} must be a scalar boundary mass flux.")
        value = value.reshape(())
        if not bool(torch.isfinite(value)):
            raise ValueError(f"{key} must be finite.")
        if bool(value < 0.0):
            raise ValueError(f"{key} must be non-negative.")
        return value

    def forward(
        # References: Richards (1931); Celia et al. (1990); van Genuchten (1980); Allen et al. (1998).
        # Mathematical expression: q_surface = (F_infiltration - F_ET) / rho_w;
        #   theta_dot = -D_h q_h + G_h u_h with q_h = -K(theta) d(h + z); mixed-form Picard
        #   solve of the implicit step; theta_new = clip(theta_raw, theta_wp, phi) with the
        #   residual routed.
        # Parameter meanings: F_infiltration and F_ET surface mass fluxes; rho_w water density;
        #   K hydraulic conductivity; h matric head; z depth; theta_wp wilting point; phi porosity.
        # Units: F [kg m-2 s-1]; q_surface, q_h, K [m s-1]; h, z [m]; theta [m3 m-3]; dt [s];
        #   water-balance residual [m].
        # Domain/constraints: one-dimensional profile within [theta_wp, phi]; dt must not exceed
        #   the stability estimate; the Picard iteration must converge or the step fails without a
        #   state update; the surface flux stays a tensor so infiltration remains differentiable.
        # Sign convention: q_surface positive downward into the column; free drainage leaves the
        #   bottom boundary.
        self,
        state: StateDict,
        time: float,
        dt: torch.Tensor | float,
        **kwargs
    ) -> dict[str, UpdateItem]:
        """
        Forward time step for soil water transport.
        
        Parameters:
            state : StateDict
                Current soil state
                Required keys: "soil_moisture" [m3 m-3]
                Required: "infiltration_flux" [kg m-2 s-1],
                "evapotranspiration_flux" [kg m-2 s-1]
                Optional: "pressure_head" [m]
            time : float
                Current time [s]
            dt : float
                Time step [s]
            
        Returns:
            new_state : StateDict
                Updated soil moisture and related variables
        """
        time_tensor = torch.as_tensor(time)
        if time_tensor.numel() != 1 or not bool(torch.isfinite(time_tensor).all()):
            raise ValueError("time must be a finite scalar.")

        dt_tensor = torch.as_tensor(dt)
        if dt_tensor.numel() != 1 or not bool(torch.isfinite(dt_tensor).all()):
            raise ValueError("dt must be a positive finite scalar.")
        dt_value = float(dt_tensor.detach().cpu().item())
        if dt_value <= 0.0:
            raise ValueError("dt must be a positive finite scalar.")

        # Get soil moisture
        soil_moisture = state.get("soil_moisture")
        if soil_moisture is None:
            raise ValueError("State must contain 'soil_moisture'.")
        if not isinstance(soil_moisture, torch.Tensor):
            raise ValueError("soil_moisture must be a torch.Tensor.")

        theta = soil_moisture.to(self.device)
        if theta.ndim != 1:
            raise ValueError("soil_moisture must be a one-dimensional profile.")
        if theta.numel() < 2:
            raise ValueError("soil_moisture must contain at least two soil layers.")
        if not bool(torch.isfinite(theta).all()):
            raise ValueError("soil_moisture must be finite.")
        lower_bound = torch.as_tensor(
            self.wilting_point,
            device=theta.device,
            dtype=theta.dtype,
        )
        upper_bound = torch.as_tensor(
            self.porosity,
            device=theta.device,
            dtype=theta.dtype,
        )
        if bool(torch.any(theta < lower_bound)) or bool(torch.any(theta > upper_bound)):
            raise ValueError(
                "soil_moisture must lie within [wilting_point, porosity]."
            )
        
        # Get pressure head or compute from soil moisture
        pressure_head = state.get("pressure_head")
        if pressure_head is None:
            pressure_head = self._compute_pressure_head(theta)
        else:
            if not isinstance(pressure_head, torch.Tensor):
                raise ValueError("pressure_head must be a torch.Tensor when supplied.")
            pressure_head = pressure_head.to(device=self.device, dtype=theta.dtype)
            if pressure_head.shape != theta.shape:
                raise ValueError("pressure_head must match the soil_moisture profile shape.")
            if not bool(torch.isfinite(pressure_head).all()):
                raise ValueError("pressure_head must be finite.")

        infiltration_flux = self._get_required_boundary_flux(
            state,
            "infiltration_flux",
            theta,
        )
        evapotranspiration_flux = self._get_required_boundary_flux(
            state,
            "evapotranspiration_flux",
            theta,
        )

        # References: Richards (1931); Celia et al. (1990); Allen et al. (1998).
        # Mathematical expression: q_surface = (F_infiltration - F_ET) / rho_w and
        # Delta theta = dt * [-D_h q_h + G_h q_surface].
        # Parameter meanings: F_infiltration and F_ET are kg m-2 s-1;
        # rho_w is kg m-3; dt is s; theta is m3 m-3.
        # Open-system route: q_surface enters through the declared surface matter port.
        water_density = theta.new_tensor(self.water_density)
        source_term = (infiltration_flux - evapotranspiration_flux) / water_density
        
        update = self._compute_iphs_richards_state_update(
            theta=theta,
            pressure_head=pressure_head,
            source_term=source_term,
            dt=dt_value,
        )
        theta_new = update["theta_new"]
        water_balance_residual = update["water_balance_residual"]
        water_flux = update["water_flux"]
        clipping_residual = update["clipping_residual"]

        iterations = torch.as_tensor(
            update["richards_iterations"],
            device=self.device,
            dtype=theta_new.dtype,
        )
        # References: Richards (1931); Celia et al. (1990).
        # Mathematical expression: theta_state = theta_(n+1);
        # q = -K(theta) * d(h + z)/dz; R_w = Delta W - integral(q_boundary dt).
        # Parameter meanings: theta_state [m3 m-3], q [m s-1], R_w [m].
        # Open-system route: surface and bottom fluxes are declared matter ports;
        # clipping and water-balance residuals remain explicit observations.
        return {
            "soil_moisture": UpdateItem(
                value=theta_new,
                mode=UpdateMode.REPLACE,
                unit="m3 m-3",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.soil_water_transport.clipping_residual": UpdateItem(
                value=clipping_residual,
                mode=UpdateMode.REPLACE,
                unit="m3 m-3",
                per_timestep=True,
                source_functor=self.name,
            ),
            "diagnostics.soil_water_transport.water_balance_residual": UpdateItem(
                value=water_balance_residual,
                mode=UpdateMode.REPLACE,
                unit="m",
                per_timestep=True,
                source_functor=self.name,
            ),
            "diagnostics.soil_water_transport.darcy_flux": UpdateItem(
                value=water_flux,
                mode=UpdateMode.REPLACE,
                unit="m s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.soil_water_transport.richards_iterations": UpdateItem(
                value=iterations,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                per_timestep=True,
                source_functor=self.name,
            ),
        }

    def _compute_iphs_richards_state_update(
        self,
        theta: Tensor,
        pressure_head: Tensor,
        source_term: Tensor,
        dt: torch.Tensor | float,
    ) -> dict[str, Tensor | float | None]:
        """Compute the canonical IPHS/DEC Richards update.

        References:
          - Richards (1931), unsaturated water balance.
          - Celia et al. (1990), mass-conservative mixed-form discretization.
          - Hirani, Nakshatrala, and Chaudhry (2008), DEC Darcy flow.
        Mathematical expression:
          theta_dot = -D_h q_h + G_h u_h,
          q_h = -star_K d(h + z), star_K = star_K^T >= 0.
        Parameter meanings:
          theta: volumetric water content [m3 m-3].
          pressure_head: matric head h [m].
          source_term: open input contribution in G_h u_h [m s-1].
          dt: time step [s].
        """

        # Compute Darcy-Hodge conductivity for the canonical transport operator.
        hydraulic_conductivity = self._compute_hydraulic_conductivity(theta)

        # Check stability constraints for the realized theta_dot update.
        dt_stable = self._check_richards_stability(
            dt, theta, pressure_head, hydraulic_conductivity
        )
        if dt > dt_stable:
            raise RuntimeError(
                f"Requested dt={dt:.6g} s exceeds the stable timestep "
                f"{dt_stable:.6g} s; the scheduler must subcycle the full step."
            )

        theta_old = theta.clone()

        # Realize theta_dot = -D_h q_h + G_h u_h with the existing Picard solver.
        theta_new, richards_iterations = self._solve_richards_equation(
            theta, pressure_head, hydraulic_conductivity, source_term, dt
        )
        if theta_new.shape != theta.shape:
            raise RuntimeError(
                "Richards solver returned a soil-moisture state with an invalid shape."
            )
        if not bool(torch.isfinite(theta_new).all()):
            raise RuntimeError(
                "Richards solver returned a non-finite soil-moisture state."
            )

        # Route saturation bounds through diagnostics instead of hidden sinks.
        # References: Celia et al. (1990), mass-conservative Richards update.
        # Mathematical expression: r_clip = theta_raw - clamp(theta_raw, theta_wp, phi).
        # Parameter meanings: theta_wp is wilting_point; phi is porosity.
        theta_new_unbounded = theta_new
        theta_new = torch.clamp(theta_new_unbounded, min=self.wilting_point, max=self.porosity)
        clipping_residual = theta_new_unbounded - theta_new

        pressure_head_new = self._compute_pressure_head(theta_new)
        hydraulic_conductivity_new = self._compute_hydraulic_conductivity(theta_new)
        water_flux = self._compute_water_flux(
            pressure_head_new,
            hydraulic_conductivity_new,
            surface_flux=source_term,
        )
        if not bool(torch.isfinite(water_flux).all()):
            raise RuntimeError("Richards solver produced a non-finite Darcy flux.")

        water_balance_residual = self._check_water_conservation(
            theta_old, theta_new, source_term, dt, water_flux
        )
        if not bool(torch.isfinite(water_balance_residual)):
            raise RuntimeError(
                "Richards solver produced a non-finite water-balance residual."
            )

        return {
            "theta_new": theta_new,
            "water_flux": water_flux,
            "water_balance_residual": water_balance_residual,
            "clipping_residual": clipping_residual,
            "richards_iterations": richards_iterations,
            "dt": dt,
        }
    
    def _compute_pressure_head(self, theta: Tensor) -> Tensor:
        """Compute pressure head from soil moisture using van Genuchten model."""
        # References: Van Genuchten (1980).
        # Equation: Se = (theta - theta_r) / (theta_s - theta_r),
        # h = -1/alpha * (Se^(-1/m) - 1)^(1/n), m = 1 - 1/n.
        # Parameters: theta [m3 m-3], theta_r [m3 m-3], theta_s [m3 m-3],
        # alpha_vg [m-1], n_vg [dimensionless].
        return van_genuchten_pressure_head(
            theta=theta,
            alpha_vg=self.alpha_vg,
            n_vg=self.n_vg,
            theta_residual=self.wilting_point,
            theta_saturated=self.porosity,
        )

    def _compute_dh_dtheta(self, theta: Tensor) -> Tensor:
        """Differentiate the canonical pressure-head relation with respect to theta.

        References: van Genuchten (1980); Celia et al. (1990).
        Mathematical expression: dh/dtheta = d[h_vG(theta)]/dtheta.
        Parameter meanings: h_vG is the canonical van Genuchten pressure head
            [m], and theta is volumetric water content [m3 m-3].
        """

        create_graph = torch.is_grad_enabled() and theta.requires_grad
        with torch.enable_grad():
            theta_for_grad = (
                theta
                if theta.requires_grad
                else theta.clone().requires_grad_(True)
            )
            pressure_head = self._compute_pressure_head(theta_for_grad)
            derivative = torch.autograd.grad(
                pressure_head,
                theta_for_grad,
                grad_outputs=torch.ones_like(pressure_head),
                create_graph=create_graph,
            )[0]
        return derivative

    def _compute_theta_from_head(self, pressure_head: Tensor) -> Tensor:
        """Compute soil moisture from pressure head."""
        # References: Van Genuchten (1980).
        # Equation: theta(h) = theta_r + (theta_s - theta_r) / (1 + (alpha*|h|)^n)^m.
        # Parameters: pressure_head [m], alpha_vg [m-1], n_vg [dimensionless],
        # theta_r [m3 m-3], theta_s [m3 m-3].
        return van_genuchten_water_retention(
            pressure_head=pressure_head,
            alpha_vg=self.alpha_vg,
            n_vg=self.n_vg,
            theta_residual=self.wilting_point,
            theta_saturated=self.porosity,
        )

    def _compute_water_capacity(self, pressure_head: Tensor) -> Tensor:
        """Compute water capacity C(h) = dtheta/dh."""
        # References: Celia et al. (1990); van Genuchten (1980).
        # Mathematical expression: C(h) = dtheta/dh using centered finite difference.
        # Parameters: pressure_head [m], epsilon_length [m], water_capacity_epsilon [dimensionless].
        step_value = self.epsilon_length
        if isinstance(step_value, Tensor):
            step_value = float(step_value.detach().mean())
        step_value = abs(float(step_value))
        step_t = torch.as_tensor(step_value, device=pressure_head.device, dtype=pressure_head.dtype)
        theta_plus = self._compute_theta_from_head(pressure_head + step_t)
        theta_minus = self._compute_theta_from_head(pressure_head - step_t)
        capacity = (theta_plus - theta_minus) / (2.0 * step_t)
        min_capacity = self.water_capacity_epsilon
        if isinstance(min_capacity, Tensor):
            min_capacity = float(min_capacity.detach().mean())
        min_capacity_value = float(min_capacity) / step_value
        return torch.clamp(torch.abs(capacity), min=min_capacity_value)
    
    def _compute_hydraulic_conductivity(self, theta: Tensor) -> Tensor:
        """Compute hydraulic conductivity from soil moisture."""
        # References: Van Genuchten (1980); Mualem (1976).
        # Equation: Se = (theta - theta_r) / (theta_s - theta_r),
        # K = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2, m = 1 - 1/n.
        # Parameters: theta [m3 m-3], theta_r [m3 m-3], theta_s [m3 m-3],
        # Ks [m s-1], n_vg [dimensionless], l [dimensionless].
        Se = effective_saturation(
            theta=theta,
            theta_residual=self.wilting_point,
            theta_saturated=self.porosity,
        )
        return van_genuchten_mualem_conductivity(
            effective_saturation=Se,
            k_saturated=self.saturated_conductivity,
            n_vg=self.n_vg,
        )
    
    def _compute_water_flux(
        self,
        pressure_head: Tensor,
        hydraulic_conductivity: Tensor,
        surface_flux: Tensor,
    ) -> Tensor:
        """Compute downward-positive interface fluxes for the column ledger.

        References:
          - Darcy (1856), Darcy-Buckingham flux law.
          - Richards (1931), vertical unsaturated water flow.
          - Celia et al. (1990), conservative mixed-form boundary ledger.
        Mathematical expression:
          q_(i+1/2) = K_(i+1/2) * (g_z - (h_(i+1)-h_i)/dz),
          K_(i+1/2) = 2 K_i K_(i+1)/(K_i + K_(i+1)).
        Parameter meanings:
          q: downward-positive volumetric water flux [m s-1].
          K: hydraulic conductivity [m s-1].
          h: pressure head [m].
          dz: layer spacing [m].
          g_z: configured gravity-head gradient [dimensionless].
          surface_flux: prescribed net surface flux [m s-1].
        Open-system route:
          q[0] is the surface matter port and q[-1] is free drainage from the
          lower boundary.
        """

        if pressure_head.ndim != 1 or pressure_head.numel() < 2:
            raise ValueError("Darcy flux requires at least two pressure-head layers.")
        if hydraulic_conductivity.shape != pressure_head.shape:
            raise ValueError(
                "Hydraulic conductivity must match the pressure-head profile shape."
            )
        surface_flux_tensor = torch.as_tensor(
            surface_flux,
            device=pressure_head.device,
            dtype=pressure_head.dtype,
        )
        if surface_flux_tensor.numel() != 1 or not bool(
            torch.isfinite(surface_flux_tensor).all()
        ):
            raise ValueError("surface_flux must be a finite scalar.")

        dz = torch.as_tensor(
            self.grid_spacing[0],
            device=pressure_head.device,
            dtype=pressure_head.dtype,
        )
        k_term = torch.as_tensor(
            hydraulic_conductivity,
            device=pressure_head.device,
            dtype=pressure_head.dtype,
        )
        harmonic_factor = torch.as_tensor(
            self.harmonic_mean_factor,
            device=pressure_head.device,
            dtype=pressure_head.dtype,
        )
        denominator_epsilon = torch.as_tensor(
            self.saturation_epsilon,
            device=pressure_head.device,
            dtype=pressure_head.dtype,
        )
        k_interface = (
            harmonic_factor
            * k_term[:-1]
            * k_term[1:]
            / (k_term[:-1] + k_term[1:] + denominator_epsilon)
        )
        head_gradient = (pressure_head[1:] - pressure_head[:-1]) / dz
        gravity_term = torch.as_tensor(
            self.gravity_head_gradient,
            device=pressure_head.device,
            dtype=pressure_head.dtype,
        )
        internal_flux = k_interface * (gravity_term - head_gradient)
        free_drainage_flux = k_term[-1] * gravity_term
        return torch.cat(
            (
                surface_flux_tensor.reshape(1),
                internal_flux,
                free_drainage_flux.reshape(1),
            )
        )
    
    def _check_richards_stability(
        self,
        dt: float,
        theta: Tensor,
        pressure_head: Tensor,
        hydraulic_conductivity: Tensor,
    ) -> float:
        """Check stability constraints for Richards equation."""
        # References: Richards (1931); Celia et al. (1990); Hillel (1998).
        # Mathematical expression: dt <= cfl * min(dz^2/(2*D), dz/|v|).
        # Parameter meanings: D [m2 s-1], v [m s-1], dz [m], cfl_number [dimensionless].
        cfl_value = self.cfl_number
        if isinstance(cfl_value, Tensor):
            cfl_value = float(cfl_value.detach())

        min_dx = min(self.grid_spacing)
        capacity = self._compute_water_capacity(pressure_head)
        diffusivity = hydraulic_conductivity / capacity

        max_diff = float(torch.max(diffusivity.detach()))
        if max_diff > 0:
            dt_diff = cfl_value * (min_dx ** 2) / (2.0 * max_diff)
        else:
            dt_diff = float("inf")

        gravity_scale = self.gravity_head_gradient
        if isinstance(gravity_scale, Tensor):
            gravity_scale = float(gravity_scale.detach())
        velocity = torch.abs(hydraulic_conductivity) * abs(float(gravity_scale))
        max_vel = float(torch.max(velocity.detach()))
        if max_vel > 0:
            dt_adv = cfl_value * min_dx / max_vel
        else:
            dt_adv = float("inf")

        dt_stable = min(dt_diff, dt_adv, dt)
        dt_stable = max(dt_stable, float(self.min_timestep))
        return dt_stable
    
    def _solve_richards_equation(
        self,
        theta: Tensor,
        pressure_head: Tensor,
        hydraulic_conductivity: Tensor,
        source_term: Tensor,
        dt: float
    ) -> Tensor:
        """
        Solve Richards equation using Picard iteration for 1D columns.

        Mathematical Expression:
            dtheta/dt = d/dz [K(theta) * (dh/dz + g_z)] + S
            where g_z is the gravity head gradient (positive downward).

        References:
            - Richards, L.A. (1931). Capillary conduction of liquids through porous mediums
            - Celia, M.A. et al. (1990). Water Resources Research, 26(7), 1483-1496
            - van Genuchten, M.Th. (1980). Soil Sci. Soc. Am. J., 44, 892-898

        Returns:
            tuple[Tensor, int]: Updated soil moisture [m3 m-3] and Picard count.
        """
        if theta.dim() != 1:
            raise ValueError("Richards solver expects 1D soil moisture profiles.")

        self.soil_layers = int(theta.shape[0])
        dz_value = self.grid_spacing[0]
        if isinstance(dz_value, Tensor):
            dz_value = float(dz_value.detach())
        self.layer_thickness = float(dz_value)

        # References: Richards (1931); Celia et al. (1990).
        # Mathematical expression: q_surface = I - ET [m s-1] enters only at
        # the upper boundary; it is not a volumetric source in every soil layer.
        # Parameter meanings: source_term is a broadcast representation of the
        # single column surface flux; all entries must therefore be identical.
        # Open-system route: surface_flux_value is the declared surface matter port.
        source_values = torch.as_tensor(
            source_term,
            device=theta.device,
            dtype=theta.dtype,
        ).reshape(-1)
        if source_values.numel() == 0 or not bool(torch.isfinite(source_values).all()):
            raise ValueError("Surface water flux must be finite and non-empty.")
        surface_flux = source_values[0]
        if not bool(torch.all(source_values == surface_flux)):
            raise ValueError(
                "The canonical surface boundary flux must be uniform per soil column."
            )
        # References: Richards (1931); Celia et al. (1990); Paszke et al. (2019), reverse-mode autograd.
        # Mathematical expression: R_0(theta) = theta_0 - theta_0,old - dt q_surface / dz + ...,
        #   so d theta_new / d q_surface is part of the solved system and must stay on the graph.
        # Parameter meanings: q_surface is the canonical net surface volumetric flux
        #   (I - ET)/rho_w carried as a 0-d tensor; the solver arithmetic is tensor-compatible.
        # Units: q_surface [m s-1]; dt [s]; dz [m]; theta [m3 m-3].
        # Domain/constraints: q_surface finite and uniform per column (validated above); it is
        #   not converted to a Python scalar, which would detach infiltration and
        #   evapotranspiration inputs from the returned soil-moisture state.
        # Sign convention: q_surface positive downward into the column.
        surface_flux_value = surface_flux.reshape(())

        tolerance = self.richards_tolerance
        if isinstance(tolerance, Tensor):
            tolerance = float(tolerance.detach())

        relax = self.under_relaxation_factor
        if isinstance(relax, Tensor):
            relax = float(relax.detach())

        max_iter = int(self.max_richards_iterations)
        theta_old = theta.clone()
        theta_iter = theta.clone()
        h_iter = pressure_head.clone()
        k_iter = hydraulic_conductivity.clone()
        converged = False
        iteration_count = 0

        for iteration in range(1, max_iter + 1):
            theta_candidate, precision_converged = (
                self._richards_solver._solve_linearized_system_picard(
                    theta_old=theta_old,
                    theta_iter=theta_iter,
                    h_iter=h_iter,
                    k_iter=k_iter,
                    dt=dt,
                    surface_flux=surface_flux_value,
                )
            )
            if relax < 1.0:
                theta_new = (
                    relax * theta_candidate + (1.0 - relax) * theta_iter
                )
            else:
                theta_new = theta_candidate
            h_new = self._compute_pressure_head(theta_new)
            k_new = self._compute_hydraulic_conductivity(theta_new)
            residual = self._richards_solver._compute_newton_residual(
                theta_new,
                theta_old,
                h_new,
                k_new,
                dt,
                surface_flux_value,
            )
            iteration_count = iteration
            if bool(torch.max(torch.abs(residual)) <= tolerance) or (
                precision_converged and relax >= 1.0
            ):
                converged = True
                break
            theta_iter = theta_new
            h_iter = h_new
            k_iter = k_new

        if not converged:
            raise RuntimeError(
                "Richards Picard iteration did not converge within the configured limit."
            )
        return theta_new, iteration_count
    
    def _check_water_conservation(
        self,
        theta_old: Tensor,
        theta_new: Tensor,
        source_term: Tensor,
        dt: float,
        water_flux: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Return the absolute column-water balance residual in meters."""
        # References: Richards (1931); Celia et al. (1990).
        # Mathematical expression: R_w = Delta W - q_surface*dt + q_bottom*dt,
        # where Delta W = sum_i[(theta_new_i - theta_old_i) dz].
        # Parameter meanings: theta [m3 m-3], dz [m], q [m s-1], dt [s].
        # Open-system route: q_surface and q_bottom are boundary matter ports;
        # R_w explicitly retains numerical and clipping imbalance.
        dz_value = getattr(self, "layer_thickness", None)
        if dz_value is None:
            dz_value = self.grid_spacing[0]
        if isinstance(dz_value, Tensor):
            dz_value = float(dz_value.detach())

        storage_change = torch.sum(theta_new - theta_old) * dz_value
        source_values = torch.as_tensor(
            source_term,
            dtype=storage_change.dtype,
            device=storage_change.device,
        ).reshape(-1)
        if source_values.numel() == 0 or not bool(torch.isfinite(source_values).all()):
            raise ValueError("Surface water flux must be finite and non-empty.")
        surface_flux = source_values[0]
        if not bool(torch.all(source_values == surface_flux)):
            raise ValueError(
                "Water balance requires one uniform surface boundary flux per column."
        )
        surface_input = surface_flux * dt
        bottom_outflow = storage_change.new_zeros(())
        if water_flux is not None and water_flux.numel() >= 1:
            bottom_outflow = water_flux.reshape(-1)[-1] * dt

        return storage_change - surface_input + bottom_outflow

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> dict[str, float]:
        """Evaluate the explicit open-column water ledger.

        References:
          - Richards (1931), vertical unsaturated water balance.
          - Celia et al. (1990), mass-conservative mixed-form discretization.
        Mathematical expression:
          R_w = sum_i[(theta_new_i - theta_old_i) dz]
                - q_surface dt + q_bottom dt.
        Parameter meanings:
          R_w is the absolute water-balance residual [m].
        Open-system route:
          Surface infiltration/evapotranspiration and bottom drainage are the
          declared matter ports; the local guard evaluates their residual.
        """

        del input_state
        residual = output_updates.get(
            "diagnostics.soil_water_transport.water_balance_residual"
        )
        if residual is None:
            return {}
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError(
                "Soil-water balance residual must be a torch.Tensor."
            )
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("Soil-water balance residual must be finite.")
        magnitude = float(torch.max(torch.abs(residual)).detach().item())
        return {
            law: magnitude
            for law in self.metadata.conservation_laws
        }


# Export
__all__ = ['SoilWaterTransportFunctor']
