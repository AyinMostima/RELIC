"""Methane transport functor including diffusion, ebullition, and plant-mediated fluxes."""

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TransportMechanismFunctor,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import StateDict
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin
from relic.utils.physical_constants import get_physical_constant


class MethaneTransportFunctor(PedosphereContractMixin, TransportMechanismFunctor):
    """Partition soil methane losses into diffusion, ebullition, and plant transport.

    Scientific references:
        - Walter and Heimann (2000) doi:10.1029/1999GB001204
        - Zhuang et al. (2004) doi:10.1029/2004GB002239
        - Chanton et al. (1992) doi:10.1029/91GB02969
        - Yue et al. (2024) doi:10.5194/gmd-17-4621-2024

    Governing equations (ASCII form):
        C = M / z
        theta_air = porosity * (1 - WFPS)
        D_eff = tortuosity * D_air * theta_air^a / porosity^b
                + D_water * WFPS
        dC/dz = (C_atm - C_soil) / z
        F_diff = -D_eff * dC/dz

        if C_mol > C_threshold:
            F_ebullition = (C_mol - C_threshold) * M_C * z / tau_eb
        else:
            F_ebullition = 0

        F_plant = k_plant * (C / z) * aerenchyma * (1 - f_root_oxid)

        F_total = F_diff + F_ebb + F_plant
        dM_transport/dt = -F_total

    This is a ParameterManager-backed phenomenological hybrid of the three
    established transport pathways. It is not an exact implementation of a
    single published transport equation and does not establish strict IPHS.
    """

    TRANSPORT_TYPE = "gas_exchange"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "pedosphere_methane_transport",
        device: Optional[torch.device] = None,
    ) -> None:
        parameter_descriptions = {
            "pedosphere.methane.transport.diffusivity_air": "Free-air methane diffusivity [m2 s-1]",
            "pedosphere.methane.transport.diffusivity_water": "Aqueous methane diffusivity [m2 s-1]",
            "pedosphere.methane.transport.tortuosity_factor": "Tortuosity factor [-]",
            "pedosphere.methane.transport.bubble_threshold_concentration": "Ebullition threshold [mol m-3]",
            "pedosphere.methane.transport.ebullition_timescale": "Characteristic ebullition timescale [s]",
            "pedosphere.methane.transport.plant_conductivity": "Aerenchyma conductivity [m2 s-1]",
            "pedosphere.methane.transport.aerenchyma_fraction": "Root aerenchyma fraction [-]",
            "pedosphere.methane.transport.root_oxidation_fraction": "Root-zone oxidation fraction [-]",
            "pedosphere.methane.transport.layer_mixing_depth": "Effective mixing depth [m]",
            "pedosphere.numerical.minimum_timestep": "Minimum timestep for methane calculations [s]",
            "pedosphere.numerical.epsilon_length": "Minimum length scale for methane transport denominators [m]",
            "pedosphere.numerical.minimum_timescale": "Minimum timescale for methane transport denominators [s]",
            "pedosphere.physical.millington_quirk_exponent_numerator": "Millington-Quirk air exponent numerator [-]",
            "pedosphere.physical.millington_quirk_exponent_denominator": "Millington-Quirk air exponent denominator [-]",
        }

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Calculate requested methane transport through diffusion, ebullition, and plant pathways",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "theta_air=porosity*(1-WFPS); D_eff=tortuosity*D_air*"
                "theta_air^a/porosity^b+D_water*WFPS; F_transport=F_diff+"
                "F_ebullition+F_plant; dM_transport/dt=-F_transport"
            ),
            references=[
                "Walter and Heimann (2000) doi:10.1029/1999GB001204",
                "Zhuang et al. (2004) doi:10.1029/2004GB002239",
                "Chanton et al. (1992) doi:10.1029/91GB02969",
                "Yue et al. (2024) doi:10.5194/gmd-17-4621-2024",
            ],
            parameters=parameter_descriptions,
            input_variables=[
                "ch4_soil_storage",
                "water_filled_pore_space",
                "porosity",
                "atmospheric_ch4_concentration",
            ],
            output_variables=[
                "ch4_transport_loss_rate",
                "ch4_storage_tendency_trans",
                "diagnostics.methane_transport.diffusion_rate",
                "diagnostics.methane_transport.ebullition_rate",
                "diagnostics.methane_transport.plant_flux_rate",
            ],
            conservation_laws=["mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            spatial_dimensions=1,
            temporal_scale=TemporalScale.SECOND,
        )
        runtime_device = device if device is not None else torch.device("cpu")
        super().__init__(name=name, metadata=metadata, device=runtime_device)

        self.diffusivity_air = self.get_parameter_value(
            "diffusivity_air",
            namespace="pedosphere.methane.transport", required=True)
        self.diffusivity_water = self.get_parameter_value(
            "diffusivity_water",
            namespace="pedosphere.methane.transport", required=True)
        self.tortuosity_factor = self.get_parameter_value(
            "tortuosity_factor",
            namespace="pedosphere.methane.transport", required=True)
        self.bubble_threshold = self.get_parameter_value(
            "bubble_threshold_concentration",
            namespace="pedosphere.methane.transport", required=True)
        self.ebullition_timescale = self.get_parameter_value(
            "ebullition_timescale",
            namespace="pedosphere.methane.transport", required=True)
        self.plant_conductivity = self.get_parameter_value(
            "plant_conductivity",
            namespace="pedosphere.methane.transport", required=True)
        self.aerenchyma_fraction = self.get_parameter_value(
            "aerenchyma_fraction",
            namespace="pedosphere.methane.transport", required=True)
        self.root_oxidation_fraction = self.get_parameter_value(
            "root_oxidation_fraction",
            namespace="pedosphere.methane.transport", required=True)
        self.layer_depth = self.get_parameter_value(
            "layer_mixing_depth",
            namespace="pedosphere.methane.transport", required=True)
        self.dt_minimum = self.get_parameter_value(
            "minimum_timestep",
            namespace="pedosphere.numerical", required=True)
        self.minimum_layer_depth = self.get_parameter_value(
            "epsilon_length",
            namespace="pedosphere.numerical", required=True)
        self.minimum_timescale = self.get_parameter_value(
            "minimum_timescale",
            namespace="pedosphere.numerical", required=True)
        self.millington_quirk_exponent_numerator = self.get_parameter_value(
            "millington_quirk_exponent_numerator",
            namespace="pedosphere.physical", required=True)
        self.millington_quirk_exponent_denominator = self.get_parameter_value(
            "millington_quirk_exponent_denominator",
            namespace="pedosphere.physical", required=True)
        self.carbon_molar_mass = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.carbon"
            )
        )

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare the methane transport cochain spaces for Phase 8 audit.

        References:
          - Walter and Heimann (2000), wetland methane transport pathways.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          x_dot = -D_h q_h + G_h u_h, q_h = -star_K d_h e_h.
        Parameter meanings:
          ch4_soil_storage is the local extensive CH4-C storage cochain,
          ch4_transport_gradient is the driving concentration gradient, and
          ch4_transport_flux is the exported matter flux through ports.
        """

        return (
            CochainSpaceSpec(
                complex_id="pedosphere_methane_transport",
                degree=0,
                primality="primal",
                n_dofs=1,
                units="g C m-2",
                extensivity="extensive",
                physical_domain="pedosphere",
                semantic_name="ch4_soil_storage",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_methane_transport",
                degree=1,
                primality="primal",
                n_dofs=1,
                units="g C m-4",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="ch4_transport_gradient",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_methane_transport",
                degree=2,
                primality="dual",
                n_dofs=1,
                units="g C m-2 s-1",
                extensivity="integral",
                physical_domain="pedosphere",
                semantic_name="ch4_transport_flux",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "pedosphere.methane.transport.diffusivity_air",
            "pedosphere.methane.transport.diffusivity_water",
            "pedosphere.methane.transport.tortuosity_factor",
            "pedosphere.methane.transport.bubble_threshold_concentration",
            "pedosphere.methane.transport.ebullition_timescale",
            "pedosphere.methane.transport.plant_conductivity",
            "pedosphere.methane.transport.aerenchyma_fraction",
            "pedosphere.methane.transport.root_oxidation_fraction",
            "pedosphere.methane.transport.layer_mixing_depth",
            "pedosphere.physical.millington_quirk_exponent_numerator",
            "pedosphere.physical.millington_quirk_exponent_denominator",
            "pedosphere.numerical.minimum_timestep",
            "pedosphere.numerical.epsilon_length",
            "pedosphere.numerical.minimum_timescale",
        )

    def build_hodge_contract(
        self,
        n_dofs: int,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the methane diffusion Hodge contract.

        References:
          - Zhuang et al. (2004), soil methane diffusion in wetlands.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = -star_K d_h e_h with star_K = diag(D_eff) and D_eff >= 0.
        Parameter meanings:
          diffusivity_air and tortuosity_factor define the passive nominal
          methane mobility used when no runtime moisture state is available.
        """
        # References: Hirani (2003); Zhuang et al. (2004).
        # Mathematical expression: star_K=diag(D_air*tau); q=-star_K*gradient.
        # Parameter meanings: D_air and tau define nominal mobility without runtime
        #   moisture; n_dofs determines the repeated diagonal size.
        # Units: star_K [m2 s-1]; gradient [g C m-4]; q [g C m-2 s-1].
        # Domain/constraints: Positive n_dofs and finite nonnegative mobility; this
        #   nominal Hodge is not the runtime moisture-dependent effective diffusivity.
        # Sign convention: q is outward positive for a negative atmosphere-minus-soil gradient.

        n_local = int(n_dofs)
        if n_local < 1:
            raise ValueError("Methane transport Hodge degrees of freedom must be positive.")
        mobility = self.diffusivity_air * self.tortuosity_factor
        if not bool(torch.all(torch.isfinite(mobility))) or bool(torch.any(mobility < 0)):
            raise ValueError("Methane transport Hodge mobility must be finite and non-negative.")
        values = mobility.reshape(1).expand(n_local).to(device=self.device)
        gradient_space = CochainSpaceSpec(
            complex_id="pedosphere_methane_transport",
            degree=1,
            primality="primal",
            n_dofs=n_local,
            units="g C m-4",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="ch4_transport_gradient",
            orientation_convention="outward_positive",
        )
        flux_space = CochainSpaceSpec(
            complex_id="pedosphere_methane_transport",
            degree=2,
            primality="dual",
            n_dofs=n_local,
            units="g C m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="ch4_transport_flux",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_methane_transport_hodge",
            domain=gradient_space,
            codomain=flux_space,
            matrix=torch.diag(values),
            parameter_keys=(
                "pedosphere.methane.transport.diffusivity_air",
                "pedosphere.methane.transport.tortuosity_factor",
            ),
            metric_keys=("pedosphere.methane.transport.layer_mixing_depth",),
            material_law="methane_fickian_boundary_transport",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build Hodge terms for Phase 8 IPHS assembly."""

        # The Hodge acts on the declared state layout unless the complex bundle
        # declares its own degrees of freedom.
        bundle_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        n_dofs = self.declared_forms()[0].n_dofs if bundle_dofs is None else bundle_dofs
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural storage-effort view for methane transport audits."""

        if "ch4_soil_storage" not in state:
            return {
                "storage_functional": "H_h = 0.5 * x_h.T x_h",
                "effort": "e_h = delta H_h / delta x_h",
            }
        storage = self._require_state_tensor(
            state,
            "ch4_soil_storage",
            lower=0.0,
        )
        return {
            "storage_functional": self._quadratic_storage(storage),
            "effort": storage,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative structural diffusion dissipation terms."""
        # References: van der Schaft and Maschke (1995); Zhuang et al. (2004).
        # Mathematical expression: sigma_struct=sum(D_eff*gradient^2).
        # Parameter meanings: gradient and D_eff are returned by the same transport
        #   operator-term helper used to describe the diffusion pathway.
        # Units: sigma_struct [g2 C2 m-6 s-1]; gradient [g C m-4]; D_eff [m2 s-1].
        # Domain/constraints: Nonnegative D_eff gives a nonnegative structural proxy;
        #   thermodynamic entropy units and full-path closure are not established here.
        # Sign convention: Squaring removes direction; this proxy is nonnegative for either flux sign.

        if "ch4_soil_storage" not in state:
            return {
                "dissipation_rate": "sigma_h = X_h.T star_K X_h",
                "thermodynamic_force": "X_h = d_h e_h",
            }
        terms = self._transport_operator_terms(state)
        return {
            "dissipation_rate": torch.sum(
                terms["effective_diffusivity"] * terms["gradient"] * terms["gradient"]
            ),
            "thermodynamic_force": terms["gradient"],
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for methane boundary transport."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive methane transport mobility contribution.

        References:
          - Zhuang et al. (2004), effective methane diffusivity.
          - van der Schaft and Maschke (1995), dissipative port-Hamiltonian form.
        Mathematical expression:
          M_h = diag(D_eff), D_eff >= 0, sigma_h = X_h.T M_h X_h >= 0.
        Parameter meanings:
          D_eff is assembled from ParameterManager-backed diffusivity,
          tortuosity, WFPS, porosity, and Millington-Quirk exponent entries.
        """
        # References: van der Schaft and Maschke (1995); Millington and Quirk (1961).
        # Mathematical expression: M=diag(D_eff), J=0; without runtime state use
        #   D_nominal=D_air*tau from build_hodge_contract.
        # Parameter meanings: D_eff is moisture-dependent diffusion mobility only;
        #   the returned state_vars name methane storage coordinates.
        # Units: M [m2 s-1]; storage coordinates [g C m-2].
        # Domain/constraints: Finite nonnegative diagonal; this structural fragment
        #   alone does not prove equivalence of the full ebullition/plant/boundary law.
        # Sign convention: The diagonal is dissipative-positive; J contains no reversible term.

        if "ch4_soil_storage" in state:
            values = self._transport_operator_terms(state)["effective_diffusivity"].reshape(-1)
        else:
            values = torch.diag(
                self.build_hodge_contract(
                    n_dofs=self.declared_forms()[0].n_dofs, strict=True
                ).matrix
            )
        if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values < 0)):
            raise ValueError("Methane transport mobility must be non-negative in M_h.")
        matrix = torch.diag(values.to(device=self.device))
        return {
            "state_vars": tuple(f"ch4_soil_storage[{idx}]" for idx in range(matrix.shape[0])),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the aggregate methane request interconnection port.

        References:
          - Walter and Heimann (2000), methane transport pathways.
          - Chanton et al. (1992), plant-mediated wetland methane transport.
        Mathematical expression:
          F_transport = F_diff + F_ebb + F_plant.
        Parameter meanings:
          F_transport is a signed request sent to MethaneBudgetCoupler, which
          owns the realized atmospheric boundary flux and storage limiter.
        """

        effort_space = CochainSpaceSpec(
            complex_id="pedosphere_methane_transport",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="g C m-3",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="ch4_transport_effort",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="pedosphere_methane_transport",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="g C m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="ch4_transport_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="methane_transport_request",
                effort_var="ch4_soil_concentration",
                flow_var="ch4_transport_loss_rate",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="g2 C2 m-5 s-1",
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
        # References: Zhuang et al. (2004); Walter and Heimann (2000); Chanton et al. (1992).
        # Mathematical expression: C=S/z; F_diff=D_eff*(C-C_atm)/z;
        #   F_eb=max(C/M_C-C_threshold,0)*M_C*z/tau_eb;
        #   F_plant=k_plant*(C/z)*aerenchyma*(1-f_root_oxid); T=F_diff+F_eb+F_plant.
        # Parameter meanings: S is areal methane carbon storage; z is mixing depth;
        #   M_C is carbon molar mass and tau_eb is the ebullition time scale.
        # Units: S [g C m-2]; C,C_atm [g C m-3]; D_eff,k_plant [m2 s-1];
        #   C_threshold [mol m-3]; M_C [g mol-1]; T and pathway rates [g C m-2 s-1].
        # Domain/constraints: Finite nonnegative storage and atmosphere, WFPS in [0,1],
        #   porosity in (0,1], positive z/tau; this is the raw transport request.
        # Sign convention: Positive T exports methane, negative diffusion imports it;
        #   tendency=-T. MethaneBudgetCoupler limits availability and advances storage once.
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        **kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        self._validate_dt(dt)
        storage = self._require_state_tensor(
            state,
            "ch4_soil_storage",
            lower=0.0,
        )
        wfps = self._require_state_tensor(
            state,
            "water_filled_pore_space",
            lower=0.0,
            upper=1.0,
        )
        porosity = self._require_state_tensor(
            state,
            "porosity",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        atmospheric_concentration = self._require_state_tensor(
            state,
            "atmospheric_ch4_concentration",
            lower=0.0,
        )

        layer_depth = self._validate_parameter_scalar(
            self.layer_depth,
            "pedosphere.methane.transport.layer_mixing_depth",
            lower=self.minimum_layer_depth,
            lower_inclusive=True,
        )
        ebullition_timescale = self._validate_parameter_scalar(
            self.ebullition_timescale,
            "pedosphere.methane.transport.ebullition_timescale",
            lower=self.minimum_timescale,
            lower_inclusive=True,
        )
        bubble_threshold = self._validate_parameter_scalar(
            self.bubble_threshold,
            "pedosphere.methane.transport.bubble_threshold_concentration",
            lower=0.0,
        )
        plant_conductivity = self._validate_parameter_scalar(
            self.plant_conductivity,
            "pedosphere.methane.transport.plant_conductivity",
            lower=0.0,
        )
        aerenchyma_fraction = self._validate_parameter_scalar(
            self.aerenchyma_fraction,
            "pedosphere.methane.transport.aerenchyma_fraction",
            lower=0.0,
            upper=1.0,
        )
        root_oxidation_fraction = self._validate_parameter_scalar(
            self.root_oxidation_fraction,
            "pedosphere.methane.transport.root_oxidation_fraction",
            lower=0.0,
            upper=1.0,
        )
        concentration = storage / layer_depth
        carbon_molar_mass = storage.new_tensor(self.carbon_molar_mass)
        concentration_mol = concentration / carbon_molar_mass

        # References: Millington and Quirk (1961); Yue et al. (2024), Eq. 42.
        # Equation: theta_air = porosity * (1 - WFPS);
        # D_eff = tortuosity * D_air * theta_air^a / porosity^b + D_water * WFPS.
        # Parameters: porosity is total pore fraction, WFPS is the water-filled
        # fraction, a and b are Millington-Quirk exponents, and tortuosity is a
        # ParameterManager-backed calibration factor. This hybrid is not the
        # complete iMAPLE Eq. 42 because soil-texture fractions are not inputs.
        effective_diffusivity = self._effective_diffusivity(wfps, porosity)
        # References: Zhuang et al. (2004) doi:10.1029/2004GB002239.
        # Equation: dC/dz = (C_atm - C_soil) / z; F_diff = -D_eff * dC/dz
        # Parameters: D_eff, C_atm, C_soil, z
        concentration_gradient = (atmospheric_concentration - concentration) / layer_depth
        diffusion_flux = -effective_diffusivity * concentration_gradient

        # References: Walter and Heimann (2000) doi:10.1029/1999GB001204
        # Equation: F_ebb = max(0, C_mol - C_thr) * M_C * z / tau_eb
        # Parameters: C_mol, C_thr, M_C, z, tau_eb
        zero = torch.zeros_like(storage)
        excess_mol = torch.maximum(concentration_mol - bubble_threshold, zero)
        ebullition_flux = (
            excess_mol * carbon_molar_mass * layer_depth / ebullition_timescale
        )

        # References: Chanton et al. (1992) doi:10.1029/91GB02969
        # Equation: F_plant = k_plant * (C / z) * aerenchyma * (1 - f_root_oxid)
        # Parameters: k_plant, C, z, aerenchyma, f_root_oxid
        gradient = concentration / layer_depth
        root_oxidation_factor = torch.ones_like(root_oxidation_fraction) - root_oxidation_fraction
        plant_flux = (
            plant_conductivity
            * gradient
            * aerenchyma_fraction
            * root_oxidation_factor
        )

        # References: Walter and Heimann (2000); Yue et al. (2024), Eq. 33.
        # Mathematical expression: F_transport = F_diff + F_ebb + F_plant.
        # Parameter meanings: F_transport is the signed raw request; the
        # MethaneBudgetCoupler owns the single storage-availability limiter and
        # the realized atmospheric boundary flux.
        total_loss = diffusion_flux + ebullition_flux + plant_flux

        updates: Dict[str, UpdateItem] = {}
        updates["diagnostics.methane_transport.diffusion_rate"] = UpdateItem(
            value=diffusion_flux,
            mode=UpdateMode.REPLACE,
            per_timestep=False,
            unit="g C m-2 s-1",
            source_functor=self.name,
        )
        updates["diagnostics.methane_transport.ebullition_rate"] = UpdateItem(
            value=ebullition_flux,
            mode=UpdateMode.REPLACE,
            per_timestep=False,
            unit="g C m-2 s-1",
            source_functor=self.name,
        )
        updates["diagnostics.methane_transport.plant_flux_rate"] = UpdateItem(
            value=plant_flux,
            mode=UpdateMode.REPLACE,
            per_timestep=False,
            unit="g C m-2 s-1",
            source_functor=self.name,
        )
        updates["ch4_transport_loss_rate"] = UpdateItem(
            value=total_loss,
            mode=UpdateMode.REPLACE,
            per_timestep=False,
            unit="g C m-2 s-1",
            source_functor=self.name,
        )
        # References: Walter and Heimann (2000) doi:10.1029/1999GB001204
        # Equation: dS/dt = P - O - T; here tendency contribution from transport is -T
        # Parameters: T (net atmospheric flux from transport pathways)
        updates["ch4_storage_tendency_trans"] = UpdateItem(
            value=-total_loss,
            mode=UpdateMode.REPLACE,
            per_timestep=False,
            unit="g C m-2 s-1",
            source_functor=self.name,
        )
        self._last_updates = updates
        return updates

    def _validate_dt(self, dt: Tensor | float) -> Tensor:
        """Return one finite positive timestep without hidden replacement."""

        dt_value = dt.to(self.device) if isinstance(dt, Tensor) else torch.as_tensor(
            dt,
            device=self.device,
        )
        if dt_value.numel() != 1 or not bool(torch.all(torch.isfinite(dt_value))):
            raise ValueError("dt must be one finite positive scalar")
        dt_value = dt_value.reshape(())
        dt_minimum = self._validate_parameter_scalar(
            self.dt_minimum,
            "pedosphere.numerical.minimum_timestep",
            lower=0.0,
            lower_inclusive=False,
        )
        if bool(dt_value < dt_minimum):
            raise ValueError("dt must be positive and not below minimum_timestep")
        return dt_value

    def _require_state_tensor(
        self,
        state: Mapping[str, Any],
        name: str,
        *,
        lower: float | None = None,
        upper: float | None = None,
        lower_inclusive: bool = True,
    ) -> Tensor:
        """Read one canonical state tensor and validate its declared domain."""

        if name not in state:
            raise KeyError(f"Missing required methane transport input: {name}")
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        tensor = value.to(self.device)
        if tensor.numel() == 0 or not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"{name} must contain finite values")
        if lower is not None:
            violates_lower = tensor < lower if lower_inclusive else tensor <= lower
            if bool(torch.any(violates_lower)):
                relation = "greater than or equal to" if lower_inclusive else "greater than"
                raise ValueError(f"{name} must be {relation} {lower}")
        if upper is not None and bool(torch.any(tensor > upper)):
            raise ValueError(f"{name} must be less than or equal to {upper}")
        return tensor

    def _validate_parameter_scalar(
        self,
        value: Any,
        name: str,
        *,
        lower: Any | None = None,
        upper: Any | None = None,
        lower_inclusive: bool = True,
    ) -> Tensor:
        """Validate a ParameterManager-backed scalar without clipping it."""

        tensor = value.to(self.device) if isinstance(value, Tensor) else torch.as_tensor(
            value,
            device=self.device,
        )
        if tensor.numel() != 1 or not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"{name} must be one finite scalar")
        tensor = tensor.reshape(())
        if lower is not None:
            lower_tensor = lower.to(self.device) if isinstance(lower, Tensor) else tensor.new_tensor(lower)
            violates_lower = tensor < lower_tensor if lower_inclusive else tensor <= lower_tensor
            if bool(violates_lower):
                qualifier = "at least" if lower_inclusive else "greater than"
                raise ValueError(f"{name} must be {qualifier} its lower bound")
        if upper is not None:
            upper_tensor = upper.to(self.device) if isinstance(upper, Tensor) else tensor.new_tensor(upper)
            if bool(tensor > upper_tensor):
                raise ValueError(f"{name} must not exceed its upper bound")
        return tensor

    def _quadratic_storage(self, storage: Tensor) -> Tensor:
        """Return the structural quadratic methane storage audit functional."""

        # References: van der Schaft and Maschke (1995); Hirani (2003).
        # Mathematical expression: H_h = 0.5 * x_h.T * x_h.
        # Parameter meanings: x_h is the methane storage cochain [g C m-2].
        vector = storage.reshape(-1)
        return vector.new_tensor(0.5) * torch.sum(vector * vector)

    def _effective_diffusivity(self, wfps: Tensor, porosity: Tensor) -> Tensor:
        """Evaluate the single methane diffusivity formula used at runtime."""
        # References: Millington and Quirk (1961); Yue et al. (2024), Eq. 42 context.
        # Mathematical expression: theta_air=phi*(1-WFPS);
        #   D_eff=tau*D_air*theta_air^a/phi^b+D_water*WFPS.
        # Parameter meanings: phi is total porosity; a/b are M-Q exponents and tau
        #   is the existing tortuosity calibration factor.
        # Units: D_eff,D_air,D_water [m2 s-1]; phi,WFPS,a,b,tau [1].
        # Domain/constraints: phi>0 and 0<=WFPS<=1 with nonnegative finite diffusivities;
        #   this hybrid lacks texture inputs and is not the complete iMAPLE Eq. 42.
        # Sign convention: Nonnegative diffusivity; flux direction comes from the gradient.

        diffusivity_air = self._validate_parameter_scalar(
            self.diffusivity_air,
            "pedosphere.methane.transport.diffusivity_air",
            lower=0.0,
        )
        diffusivity_water = self._validate_parameter_scalar(
            self.diffusivity_water,
            "pedosphere.methane.transport.diffusivity_water",
            lower=0.0,
        )
        tortuosity_factor = self._validate_parameter_scalar(
            self.tortuosity_factor,
            "pedosphere.methane.transport.tortuosity_factor",
            lower=0.0,
        )
        mq_numerator = self._validate_parameter_scalar(
            self.millington_quirk_exponent_numerator,
            "pedosphere.physical.millington_quirk_exponent_numerator",
            lower=0.0,
            lower_inclusive=False,
        )
        mq_denominator = self._validate_parameter_scalar(
            self.millington_quirk_exponent_denominator,
            "pedosphere.physical.millington_quirk_exponent_denominator",
            lower=0.0,
            lower_inclusive=False,
        )
        parameter_tensors = (
            diffusivity_air,
            diffusivity_water,
            tortuosity_factor,
            mq_numerator,
            mq_denominator,
        )
        (
            diffusivity_air,
            diffusivity_water,
            tortuosity_factor,
            mq_numerator,
            mq_denominator,
        ) = tuple(
            parameter.to(device=porosity.device, dtype=porosity.dtype)
            for parameter in parameter_tensors
        )

        # References: Millington and Quirk (1961); Yue et al. (2024), Eq. 42.
        # Mathematical expression: theta_air = phi * (1 - WFPS);
        # D_eff = tau * D_air * theta_air^a / phi^b + D_water * WFPS.
        # Parameter meanings: phi is total porosity, WFPS is water-filled pore
        # space, tau is the calibration factor, and a/b are the M-Q exponents.
        air_filled_porosity = porosity * (torch.ones_like(wfps) - wfps)
        air_component = (
            diffusivity_air
            * torch.pow(air_filled_porosity, mq_numerator)
            / torch.pow(porosity, mq_denominator)
        )
        water_component = diffusivity_water * wfps
        effective_diffusivity = tortuosity_factor * air_component + water_component
        if not bool(torch.all(torch.isfinite(effective_diffusivity))) or bool(
            torch.any(effective_diffusivity < 0)
        ):
            raise ValueError(
                "Methane transport effective diffusivity must be finite and non-negative"
            )
        return effective_diffusivity

    def _transport_operator_terms(self, state: Mapping[str, Any]) -> Mapping[str, Tensor]:
        """Assemble ParameterManager-backed methane transport operator terms."""
        # References: Zhuang et al. (2004); Millington and Quirk (1961).
        # Mathematical expression: C=S/z; gradient=(C_atm-C)/z;
        #   D_eff=_effective_diffusivity(WFPS,phi).
        # Parameter meanings: S,z,C_atm,WFPS and phi are the same governed inputs
        #   and coefficients as the runtime transport request.
        # Units: S [g C m-2]; C [g C m-3]; gradient [g C m-4]; D_eff [m2 s-1].
        # Domain/constraints: This assembles structural audit terms and delegates the
        #   single diffusivity law; it does not include ebullition or plant transport.
        # Sign convention: The chosen gradient is atmosphere minus soil; outward F=-D*gradient.

        storage = self._require_state_tensor(
            state,
            "ch4_soil_storage",
            lower=0.0,
        )
        wfps = self._require_state_tensor(
            state,
            "water_filled_pore_space",
            lower=0.0,
            upper=1.0,
        )
        porosity = self._require_state_tensor(
            state,
            "porosity",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        atmospheric_concentration = self._require_state_tensor(
            state,
            "atmospheric_ch4_concentration",
            lower=0.0,
        )
        layer_depth = self._validate_parameter_scalar(
            self.layer_depth,
            "pedosphere.methane.transport.layer_mixing_depth",
            lower=self.minimum_layer_depth,
        )
        concentration = storage / layer_depth

        effective_diffusivity = self._effective_diffusivity(wfps, porosity)
        gradient = (atmospheric_concentration - concentration) / layer_depth
        return {
            "storage": storage,
            "wfps": wfps,
            "air_porosity": porosity * (torch.ones_like(wfps) - wfps),
            "porosity": porosity,
            "concentration": concentration,
            "effective_diffusivity": effective_diffusivity.reshape(-1),
            "gradient": gradient.reshape(-1),
        }

    def _update_tensor(self, value: Any) -> Tensor:
        """Unwrap one update value for full-tensor conservation checks."""

        if isinstance(value, UpdateItem):
            value = value.value
        if not isinstance(value, Tensor):
            raise TypeError("Methane transport conservation values must be tensors")
        return value.to(self.device)

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """
        Check methane transport closure against the net storage tendency update.

        References:
            - Walter and Heimann (2000) doi:10.1029/1999GB001204
            - Yue et al. (2024) doi:10.5194/gmd-17-4621-2024
        Mathematical expression:
            violation["mass"] = |delta_T + delta_dS_trans_dt|
        Parameter meanings:
            - ch4_transport_loss_rate: update applied to the net atmospheric transport loss [g C m-2 s-1]
            - ch4_storage_tendency_trans: transport contribution to storage tendency [g C m-2 s-1]
        """
        # References: Walter and Heimann (2000); Yue et al. (2024), methane storage budget.
        # Mathematical expression: residual=max(abs(T+storage_tendency_trans)).
        # Parameter meanings: T is the signed raw transport request and the tendency
        #   is its matching opposite port, both extracted from public UpdateItems.
        # Units: residual and both terms [g C m-2 s-1].
        # Domain/constraints: Finite nonempty tensor residual; absent ports give no
        #   result. This rate identity does not verify the coupler's realized storage step.
        # Sign convention: Positive export matches negative storage tendency; import reverses both.
        if "mass" not in self.metadata.conservation_laws:
            return {}

        transport_update = output_updates.get("ch4_transport_loss_rate")
        tendency_update = output_updates.get("ch4_storage_tendency_trans")
        if transport_update is None or tendency_update is None:
            return {}

        transport_value = self._update_tensor(transport_update)
        tendency_value = self._update_tensor(tendency_update)
        residual = torch.abs(transport_value + tendency_value)
        if residual.numel() == 0 or not bool(torch.all(torch.isfinite(residual))):
            raise ValueError("Methane transport conservation residual must be finite")
        return {"mass": float(torch.max(residual).detach().cpu().item())}


__all__ = ["MethaneTransportFunctor"]

