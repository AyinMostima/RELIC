"""
Capillary Rise Transport Module for RELIC

This module implements capillary rise transport based on Gardner (1958) 
and Eagleson (1978) analytical approximations for steady-state upward
water transport from groundwater table to soil surface.

References
----------
- Gardner, W.R. (1958). Some steady-state solutions of the unsaturated 
  moisture flow equation with application to evaporation from a water table.
  Soil Science, 85(4), 228-232.
- Eagleson, P.S. (1978). Climate, soil, and vegetation: A reduced-order model 
  of soil moisture movement in the liquid phase. Water Resources Research, 
  14(5), 722-730.
- RHESSys compute_capillary_rise.c implementation

Mathematical Expression
------------------------
Steady-state Richards equation:
dT/dt = 0 = d/dz[K(T)(dh/dz - 1)]

Gardner approximation for capillary rise flux:
w = w_max = B*a/(Z^b)
where:
- w = capillary rise flux [m s-1]  
- Z = water table depth [m]
- B = 1 + 1.5/(b-1) (Gardner coefficient)
- a, b = hydraulic parameters from K(psi) = a * psi^b

Units Convention:
- capillary_rise_flux: kg m-2 s-1 (mass flux)
- water_table_depth: m (positive downward)
- hydraulic_conductivity: m s-1
"""

from typing import Any, Dict, Mapping, Optional, Tuple, Union
import torch
from torch import Tensor

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import FluxDict, StateDict
from relic.utils.physical_constants import get_physical_constant
from relic.utils.soil_hydraulics import van_genuchten_m_parameter
from ..interface_mapping import HydrosphereInterfaceMixin
from ..solvers.richards_solver import on_device_of



class CapillaryRiseTransport(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Capillary Rise Transport Mechanism
    
    Estimates upward water transport from groundwater to soil layers
    using Gardner-Eagleson steady-state analytical approximation.
    Provides coupling between groundwater dynamics and soil moisture.
    
    This implementation handles:
    - Steady-state capillary rise flux calculation
    - Depth-dependent porosity profiles n(z) = n0 * exp(-z/p)
    - Multiple hydraulic parameterizations (Brooks-Corey, Van Genuchten)
    - Maximum capillary height limitations
    - Integration with Richards equation as boundary condition
    
    Attributes
    ----------
    max_capillary_height : float
        Maximum height of capillary fringe [m]
    pore_size_index : float
        Brooks-Corey pore size distribution index [dimensionless]
    air_entry_pressure : float
        Brooks-Corey air entry pressure head [m]
    surface_porosity : float
        Porosity at soil surface [dimensionless]
    porosity_decay : float
        Exponential porosity decay parameter [m-1]
    """

    TRANSPORT_TYPE = "capillary_rise"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    
    def __init__(self, 
                 name: str = "capillary_rise_transport",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize capillary rise transport mechanism.
        
        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Configuration parameters for capillary rise properties
        """
        # User config is passed through to parent class
        if config is None:
            config = {}
            
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Capillary rise transport from groundwater using Gardner-Eagleson approximation",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "m_dot = G_cap u_cap; u_cap = rho_w * min(Ks, B*a/Z^b); "
                "B = 1 + 1.5/(b-1)"
            ),
            references=[
                "Gardner, W.R. (1958). Soil Science, 85(4), 228-232",
                "Eagleson, P.S. (1978). Water Resources Research, 14(5), 722-730",
                "RHESSys compute_capillary_rise.c"
            ],
            parameters={
                "hydrosphere.transport.max_capillary_height": (
                    "Maximum height of capillary fringe (water table depth limit for significant "
                    "capillary rise) [m]"
                ),
                "hydrosphere.transport.brooks_corey_pore_size_index": (
                    "Brooks-Corey pore size distribution index (lambda) for capillary rise "
                    "calculations [dimensionless]"
                ),
                "hydrosphere.transport.air_entry_pressure": (
                    "Brooks-Corey air entry pressure head (water column) for capillary rise "
                    "calculations [m]"
                ),
                "hydrosphere.groundwater.surface_porosity": (
                    "Porosity at soil surface for exponential decay profile n(z) = n0 * exp(-z/p) "
                    "[dimensionless]"
                ),
                "hydrosphere.groundwater.porosity_decay_length": (
                    "Exponential decay length scale p in porosity profile n(z) = n0 * exp(-z/p) [m]"
                ),
                "hydrosphere.transport.capillary_saturation_threshold_factor": (
                    "Fraction of surface porosity defining saturation threshold for capillary rise "
                    "[dimensionless]"
                ),
                "hydrosphere.transport.capillary_saturation_bandwidth_factor": (
                    "Saturation bandwidth factor for capillary rise reduction near saturation "
                    "[dimensionless]"
                ),
                "hydrosphere.transport.hydraulic_parameterization_method": (
                    "Hydraulic parameterization method (brooks_corey or van_genuchten) "
                    "[dimensionless]"
                ),
                "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (
                    "Saturated hydraulic conductivity for Richards equation [m s-1]"
                ),
                "hydrosphere.soil_moisture.alpha_vg": (
                    "Van Genuchten alpha parameter (inverse air entry pressure) [m-1]"
                ),
                "hydrosphere.soil_moisture.n_vg": (
                    "Van Genuchten n parameter (pore size distribution index) [dimensionless]"
                ),
                "hydrosphere.transport.enable_capillary_depth_profile": (
                    "Enable depth-dependent capillary rise flux profile calculation [dimensionless]"
                ),
                "hydrosphere.transport.capillary_profile_layer_count": (
                    "Default number of soil layers for capillary rise flux profile when not "
                    "provided by caller [dimensionless]"
                ),
                "hydrosphere.transport.capillary_profile_layer_thickness": (
                    "Default layer thickness for capillary rise profile when not provided by caller"
                    " [m]"
                ),
            },
            input_variables=[
                "water_table_depth",
                "soil_moisture_profile",
            ],
            output_variables=[
                "capillary_rise_flux",
                "diagnostics.capillary_rise",
                "diagnostics.capillary_rise.conductivity_clipping_residual",
                "diagnostics.capillary_rise.saturation_constraint_residual",
            ],
            conservation_laws=["mass"],
            spatial_dimensions=1,  # 1D vertical profile
            temporal_scale=TemporalScale.STEADY_STATE,
            coupling_interfaces=["soil_moisture", "groundwater", "evapotranspiration"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        
        # Initialize parent classes (the label is the class attribute TRANSPORT_TYPE)
        super().__init__(name=name, config=config, metadata=metadata)
        self.optional_input_variables = {"soil_moisture_profile"}
        
        # Add additional TransportMechanism attributes
        # The Gardner steady-state flux q = B a / Z^b is an algebraic flux law without a
        # scalar diffusivity (None) and without tortuosity.
        self.peclet_number = None
        self.velocity_field = None
        
        # Get parameters from parameter library with full namespace
        self.max_capillary_height = self.get_parameter_value(
            "max_capillary_height",
            namespace="hydrosphere.transport", required=True)
        self.pore_size_index = self.get_parameter_value(
            "brooks_corey_pore_size_index",
            namespace="hydrosphere.transport", required=True)
        self.air_entry_pressure = self.get_parameter_value(
            "air_entry_pressure",
            namespace="hydrosphere.transport", required=True)
        self.surface_porosity = self.get_parameter_value(
            "surface_porosity",
            namespace="hydrosphere.groundwater", required=True)
        self.capillary_saturation_threshold_factor = self.get_parameter_value(
            "capillary_saturation_threshold_factor",
            namespace="hydrosphere.transport", required=True)
        self.capillary_saturation_bandwidth_factor = self.get_parameter_value(
            "capillary_saturation_bandwidth_factor",
            namespace="hydrosphere.transport", required=True)
        porosity_decay_length = self.get_parameter_value(
            "porosity_decay_length",
            namespace="hydrosphere.groundwater", required=True)
        # References: Eagleson (1978); RHESSys porosity profile implementation.
        # Formula: porosity_decay = 1 / porosity_decay_length.
        # Parameters: porosity_decay_length [m], porosity_decay [m-1].
        self.porosity_decay = 1.0 / float(porosity_decay_length)
        self.hydraulic_method = self.get_parameter_value(
            "hydraulic_parameterization_method",
            namespace="hydrosphere.transport", required=True)
        self.k_saturated = self.get_parameter_value(
            "saturated_hydraulic_conductivity",
            namespace="hydrosphere.soil_moisture", required=True)
        self.alpha_vg = self.get_parameter_value(
            "alpha_vg",
            namespace="hydrosphere.soil_moisture", required=True)
        self.n_vg = self.get_parameter_value(
            "n_vg",
            namespace="hydrosphere.soil_moisture", required=True)
        self.water_density = get_physical_constant("water_density")
        self.enable_depth_profile = self.get_parameter_value(
            "enable_capillary_depth_profile",
            namespace="hydrosphere.transport", required=True)
        self.capillary_profile_layer_count = int(
            self.get_parameter_value(
                "capillary_profile_layer_count",
                namespace="hydrosphere.transport", required=True)
        )
        self.capillary_profile_layer_thickness = float(
            self.get_parameter_value(
                "capillary_profile_layer_thickness",
                namespace="hydrosphere.transport", required=True)
        )
        # m_vg and the Gardner coefficients a, b and B are properties derived
        # from the held values at every read; nothing derived from a learnable
        # binding is stored on the instance. The closure domain (b > 1) is
        # checked here once for the construction values.
        self._compute_gardner_coefficients()

        # Validate parameters
        self._validate_parameters()

    @property
    def m_vg(self) -> Tensor:
        """van Genuchten exponent m [1] from the current n value."""
        # References: van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898
        #   (Mualem restriction m = 1 - 1/n).
        # Mathematical expression: m = 1 - 1 / n.
        # Parameter meanings: n is the pore-size distribution index
        #   (hydrosphere.soil_moisture.n_vg); m the derived exponent of the van
        #   Genuchten form of the Gardner exponent, b = 2 / m.
        # Units: n and m [1].
        # Domain/constraints: n > 1, so 0 < m < 1. n is a learnable binding held
        #   by reference, so m is derived at every read and never stored: it
        #   follows a changed n, and no autograd graph is carried from one call
        #   into the next.
        # Sign convention: none (a dimensionless shape exponent).
        return van_genuchten_m_parameter(self.n_vg)

    # References: Gardner (1958); Eagleson (1978); Clapp and Hornberger (1978).
    # Formula: K(psi) = Ks * (psi_ae/psi)^(2 + 3*lambda); B = 1 + 1.5 / (b - 1).
    # Parameters: Ks [m s-1], psi_ae [m], lambda (pore_size_index) [dimensionless], b [dimensionless].
    def _compute_gardner_coefficients(
        self, reference: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the Gardner coefficients a, b and B from the current held values.
        
        For Brooks-Corey:
        K(psi) = Ks * (psi_ae/psi)^(2 + 3*lambda) for psi < psi_ae
        psi(Se) = psi_ae * Se^(-1/lambda)
        
        For Van Genuchten-Mualem: 
        Approximate with equivalent Brooks-Corey parameters

        Parameters
        ----------
        reference : Tensor, optional
            State-side operand the coefficients are combined with. The held
            values are read on its device (values and dtypes unchanged), so the
            coefficients are computed on the device of the state whatever
            device the bindings sit on. Without it they have the device of the
            held values.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            (a, b, B) of the flux law q = B * a / Z^b. They are derived at every
            call and never stored: Ks, the air-entry head, the pore-size index,
            alpha and n are learnable ParameterManager bindings held by
            reference, so a stored coefficient would keep its construction value
            after a binding changed and would carry its autograd graph into the
            next call.

        Raises
        ------
        ValueError
            If the exponent b is not finite or not greater than 1.
        """
        # References:
        #   Gardner (1958); Eagleson (1978); retained Brooks-Corey/van Genuchten approximation.
        # Mathematical expression:
        #   Brooks-Corey: b=2+3*lambda_p, a=Ks*abs(h_entry)^b;
        #   van Genuchten approximation: b=2/m, a=Ks*(1/alpha)^b;
        #   B=1+1.5/(b-1), defined for b > 1; q=B*a/Z^b.
        # Parameter meanings:
        #   lambda_p is the pore-size index; alpha/m are van Genuchten controls;
        #   h_entry is air-entry head. All closures use the same limited flux owner.
        # Units:
        #   h_entry/Z [m], alpha [m-1], Ks/q [m s-1], b/B/m/lambda_p [1];
        #   a [m^(b+1) s-1].
        # Domain/constraints:
        #   The van Genuchten relation here is the documented reduced-order
        #   Gardner approximation, not an exact Richards solution. Invalid distance
        #   support returns zero and conductivity/projection residuals are explicit.
        #   A held binding has the device of the latest registration of any
        #   Functor holding it; it is read here on the device of the reference
        #   operand (on_device_of), never moved.
        # Sign convention:
        #   q is an upward nonnegative magnitude on valid unsaturated support.
        k_saturated = on_device_of(self.k_saturated, reference)
        if self.hydraulic_method == "brooks_corey":
            # Direct Brooks-Corey parameterization
            # b = 2 + 3*lambda where lambda is pore size index
            gardner_b = 2.0 + 3.0 * on_device_of(self.pore_size_index, reference)
            
            # a = Ks * |psi_ae|^b 
            air_entry_pressure = on_device_of(self.air_entry_pressure, reference)
            gardner_a = k_saturated * (abs(air_entry_pressure) ** gardner_b)
            
        else:  # van_genuchten
            # Approximate Van Genuchten with equivalent Brooks-Corey
            # Use empirical relationships from literature
            # Approximate b value for VG
            m_vg = van_genuchten_m_parameter(on_device_of(self.n_vg, reference))
            gardner_b = 2.0 / m_vg  # Reduced-order approximation
            
            # Scale factor based on VG parameters
            characteristic_pressure = 1.0 / on_device_of(self.alpha_vg, reference)  # m
            gardner_a = k_saturated * (characteristic_pressure ** gardner_b)
        
        # Gardner B coefficient: B = 1 + 1.5/(b-1). Every valid closure gives b > 1
        # (Brooks-Corey b = 2 + 3 lambda_p with lambda_p > 0; van Genuchten b = 2/m with
        # 0 < m < 1), so b <= 1 is an invalid parameter set and raises; no limiting-case
        # default replaces B.
        exponent = torch.as_tensor(gardner_b)
        if not bool(torch.all(torch.isfinite(exponent) & (exponent > 1.0))):
            raise ValueError(
                f"Gardner exponent b must exceed 1 for B = 1 + 1.5/(b - 1), got {gardner_b}"
            )
        gardner_B = 1.0 + 1.5 / (gardner_b - 1.0)
        return gardner_a, gardner_b, gardner_B

    @property
    def gardner_a(self) -> Tensor:
        """Gardner coefficient a [m^(b+1) s-1] from the current held values."""
        return self._compute_gardner_coefficients()[0]

    @property
    def gardner_b(self) -> Tensor:
        """Gardner exponent b [1] from the current held values."""
        return self._compute_gardner_coefficients()[1]

    @property
    def gardner_B(self) -> Tensor:
        """Gardner coefficient B = 1 + 1.5/(b - 1) [1] from the current held values."""
        return self._compute_gardner_coefficients()[2]
    
    def _validate_parameters(self) -> None:
        """Validate parameter consistency and physical constraints."""
        def _to_scalar(value) -> float:
            if isinstance(value, Tensor):
                return float(value.detach())
            return float(value)

        max_capillary_height = _to_scalar(self.max_capillary_height)
        pore_size_index = _to_scalar(self.pore_size_index)
        air_entry_pressure = _to_scalar(self.air_entry_pressure)
        surface_porosity = _to_scalar(self.surface_porosity)
        porosity_decay = _to_scalar(self.porosity_decay)
        k_saturated = _to_scalar(self.k_saturated)

        if max_capillary_height <= 0:
            raise ValueError("Maximum capillary height must be positive")
        
        if pore_size_index <= 0:
            raise ValueError("Pore size index must be positive")
        
        if air_entry_pressure >= 0:
            raise ValueError("Air entry pressure must be negative")
        
        if not (0 < surface_porosity <= 1):
            raise ValueError("Surface porosity must be in range (0, 1]")
        
        if porosity_decay < 0:
            raise ValueError("Porosity decay parameter must be non-negative")
            
        if k_saturated <= 0:
            raise ValueError("Saturated hydraulic conductivity must be positive")

    @staticmethod
    def _scalar_from_value(value: Union[Tensor, float, int]) -> float:
        """Convert a scalar tensor or numeric value to float for structural audits."""

        if isinstance(value, Tensor):
            flattened = value.detach().reshape(-1)
            if flattened.numel() == 0:
                raise ValueError("Cannot convert an empty tensor to scalar.")
            scalar_value = flattened[0].tolist()
            return float(scalar_value)
        return float(value)

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve the structural audit size without changing runtime tensor shape."""

        if n_dofs is not None:
            value = int(n_dofs)
        elif complex_bundle is not None and hasattr(complex_bundle, "n_dofs"):
            value = int(getattr(complex_bundle, "n_dofs"))
        else:
            value = int(self.capillary_profile_layer_count)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for capillary rise transport.

        References:
          - Gardner (1958), steady capillary rise from a water table.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          m_dot = G_cap u_cap, u_cap = rho_w * min(Ks, B*a/Z^b).
        Parameter meanings:
          Z is groundwater depth [m]; Ks is saturated conductivity [m s-1];
          rho_w is water density [kg m-3].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_capillary_rise",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="m",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="water_table_depth",
                orientation_convention="upward_positive",
                boundary_role="boundary",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_capillary_rise",
                degree=1,
                primality="primal",
                n_dofs=n_dofs,
                units="m s-1",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="capillary_volumetric_flux",
                orientation_convention="upward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_capillary_rise",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="capillary_mass_flux",
                orientation_convention="upward_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.transport.max_capillary_height",
            "hydrosphere.transport.brooks_corey_pore_size_index",
            "hydrosphere.transport.air_entry_pressure",
            "hydrosphere.groundwater.surface_porosity",
            "hydrosphere.groundwater.porosity_decay_length",
            "hydrosphere.transport.capillary_saturation_threshold_factor",
            "hydrosphere.transport.capillary_saturation_bandwidth_factor",
            "hydrosphere.transport.hydraulic_parameterization_method",
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
            "hydrosphere.soil_moisture.alpha_vg",
            "hydrosphere.soil_moisture.n_vg",
            "hydrosphere.transport.enable_capillary_depth_profile",
            "hydrosphere.transport.capillary_profile_layer_count",
            "hydrosphere.transport.capillary_profile_layer_thickness",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the capillary volumetric-to-mass flux Hodge contract.

        References:
          - Gardner (1958), capillary rise flux approximation.
          - Hirani (2003), material Hodge star between primal and dual cochains.
          - van der Schaft and Maschke (1995), dissipative port variables.
        Mathematical expression:
          u_cap = star_cap u_vol, star_cap = diag(rho_w), u_vol = max(0, min(Ks, B*a/Z^b)).
        Parameter meanings:
          u_vol is the volumetric capillary flux cochain produced by the runtime
          Gardner law with its Ks cap; u_cap is the mass flux cochain on the port;
          rho_w is liquid water density. Ks enters u_vol only, never star_cap.
        Units:
          star_cap entries [kg m-3]; u_vol [m s-1]; u_cap [kg m-2 s-1];
          operator_units * domain.units = codomain.units.
        Domain/constraints:
          rho_w > 0 so star_cap is symmetric positive definite; n_dofs >= 1.
        Sign convention:
          upward-positive on both domain and codomain cochains.
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        device = getattr(self, "device", torch.device("cpu"))
        conversion_value = self._scalar_from_value(self.water_density)
        values = torch.full((n_dofs,), conversion_value, dtype=torch.float32, device=device)
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Water density must be positive for capillary Hodge assembly.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_capillary_rise",
            degree=1,
            primality="primal",
            n_dofs=n_dofs,
            units="m s-1",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="capillary_volumetric_flux",
            orientation_convention="upward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_capillary_rise",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="capillary_mass_flux",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_capillary_mobility_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=("relic.physical.constants.water.density_liquid",),
            metric_keys=("hydrosphere.transport.capillary_profile_layer_thickness",),
            material_law="gardner_capillary_rise",
            operator_units="kg m-3",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build capillary Hodge terms for IPHS assembly."""

        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
            ),
        )

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
        """Return no reversible interconnection for capillary boundary transport."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive capillary conversion contribution.

        References:
          - Gardner (1958), upward capillary flux.
          - van der Schaft and Maschke (1995), dissipative port-Hamiltonian form.
        Mathematical expression:
          M_h = star_cap = diag(rho_w), sigma = u_vol.T M_h u_vol >= 0.
        Parameter meanings:
          u_vol is the volumetric capillary flux cochain; rho_w is liquid water
          density; M_h is the same conversion Hodge as build_hodge_contract.
        Units:
          M_h [kg m-3]; u_vol [m s-1]; sigma [kg m-1 s-2].
        Domain/constraints:
          rho_w > 0 gives a symmetric positive definite M_h; the Gardner Ks cap
          is applied to u_vol at runtime and is not a linear mobility in M_h.
        Sign convention:
          sigma >= 0 is a positivity audit of the upward-positive flux cochain.
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"capillary_water_storage[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the groundwater-to-soil capillary matter port.

        References:
          - Richards (1931), boundary flux form of unsaturated flow.
          - Gardner (1958), steady water-table capillary flux.
        Mathematical expression:
          P_port = e_h * f_h, with e_h = water-table head [m] and
          f_h = capillary mass flux [kg m-2 s-1].
        Parameter meanings:
          water_table_depth supplies the boundary effort; capillary_rise_flux is the flow.
        """

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_capillary_rise",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="water_table_boundary_head",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_capillary_rise",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="capillary_boundary_mass_flux",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="capillary_groundwater_boundary_flux",
                effort_var="water_table_boundary_head",
                flow_var="capillary_rise_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="kg m-1 s-1",
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
    
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return list of dependencies for this functor.
        
        Returns
        -------
        DependencyDeclaration
            Lagged groundwater owner dependency. The accepted start-of-step
            water-table state drives the current capillary transfer.
        """
        return self._merge_declared_dependencies({
            "required": [],
            "optional": [],
            "lagged": ["groundwater_dynamics"],
        })
    
    def forward(self, 
        # References:
        #   Gardner (1958); Eagleson (1978).
        # Mathematical expression:
        #   qsurface=min(max(B*a/Z^b,0),Ks) on 0<Z<=Zmax, otherwise 0;
        #   Fsurface=rho_w*qsurface.
        # Parameter meanings:
        #   Z is water-table depth below the surface; a,b,B are the retained
        #   Gardner approximation coefficients and Ks caps conductivity.
        # Units:
        #   Z/Zmax [m], qsurface/Ks [m s-1], Fsurface [kg m-2 s-1];
        #   a [m^(b+1) s-1], b/B [1], rho_w [kg m-3].
        # Domain/constraints:
        #   Finite nonnegative water-table depth; output is an instantaneous REPLACE
        #   rate with no dt integration. Optional profile saturation limits affect
        #   the diagnostic profile only, not the primary surface boundary value.
        # Sign convention:
        #   Positive primary flux rises from groundwater toward the soil surface.
                state: StateDict, 
                time: float, 
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute capillary rise flux using Gardner-Eagleson approximation.
        
        Parameters
        ----------
        state : StateDict
            Current system state containing groundwater depth
        time : float
            Current simulation time [s] (not used in steady-state)
        dt : float  
            Time step [s] (not used in steady-state)
        **kwargs
            Additional parameters and soil layer information
            
        Returns
        -------
        FluxDict
            Capillary rise flux and diagnostic variables
            
        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        state = self._validate_state_inputs(state)
        water_table_depth = self._ensure_tensor(state["water_table_depth"]).to(
            device=getattr(self, "device", torch.device("cpu"))
        )  # m

        # Validate water-table depth
        if not torch.isfinite(water_table_depth).all():
            raise ValueError("Water table depth must contain only finite values")
        if torch.any(water_table_depth < 0):
            raise ValueError("Water table depth must be positive (depth below surface)")

        return self._compute_iphs_capillary_rise_update(
            state=state,
            water_table_depth=water_table_depth,
            time=time,
            dt=dt,
            kwargs=kwargs,
        )

    def _compute_iphs_capillary_rise_update(
        self,
        state: StateDict,
        water_table_depth: Tensor,
        time: float,
        dt: float,
        kwargs: Mapping[str, Any],
    ) -> FluxDict:
        """Compute the canonical IPHS/DEC capillary boundary update.

        References:
          - Gardner (1958), steady capillary rise from a water table.
          - Eagleson (1978), reduced-order vertical liquid water movement.
          - Hirani (2003), DEC boundary flux representation.
        Mathematical expression:
          m_dot = G_cap u_cap, u_cap = rho_w * min(Ks, B*a/Z^b).
        Parameter meanings:
          Z is groundwater depth [m]; a, b, and B are Gardner coefficients;
          Ks is saturated hydraulic conductivity [m s-1]; rho_w is water density.
        Units:
          Boundary flux and limiter residuals use kg m-2 s-1.
        Open-system route:
          capillary_rise_flux is the producer-side groundwater boundary export
          and the Richards soil-boundary input. The
          conductivity residual closes the candidate-versus-retained flux ledger.
          The saturation residual compares diagnostic q(z) profiles only; neither
          residual is applied as a water update.
        """

        # References:
        #   Gardner (1958); Eagleson (1978).
        # Mathematical expression:
        #   qsurface=min(max(B*a/Z^b,0),Ks) on 0<Z<=Zmax, otherwise 0;
        #   Fsurface=rho_w*qsurface.
        # Parameter meanings:
        #   Z is water-table depth below the surface; a,b,B are the retained
        #   Gardner approximation coefficients and Ks caps conductivity.
        # Units:
        #   Z/Zmax [m], qsurface/Ks [m s-1], Fsurface [kg m-2 s-1];
        #   a [m^(b+1) s-1], b/B [1], rho_w [kg m-3].
        # Domain/constraints:
        #   Finite nonnegative water-table depth; output is an instantaneous REPLACE
        #   rate with no dt integration. Optional profile saturation limits affect
        #   the diagnostic profile only, not the primary surface boundary value.
        # Sign convention:
        #   Positive primary flux rises from groundwater toward the soil surface.
        max_height = self._ensure_tensor(self.max_capillary_height).to(
            device=water_table_depth.device, dtype=water_table_depth.dtype
        )
        # Profile layer count: the call-time soil_layers override or the configured
        # capillary_profile_layer_count.
        soil_layers = int(kwargs.get("soil_layers", self.capillary_profile_layer_count))
        # The optional soil_moisture_profile must describe the same layers as the flux
        # profile; another size is an input error on every path, not a reason to skip
        # the saturation limit.
        if "soil_moisture_profile" in state:
            profile_size = self._ensure_tensor(state["soil_moisture_profile"]).numel()
            if profile_size != soil_layers:
                raise ValueError(
                    f"soil_moisture_profile has {profile_size} values but the "
                    f"capillary flux profile has {soil_layers} layers"
                )
        if torch.all(water_table_depth > max_height):
            return self._zero_flux_result(water_table_depth)

        layer_thickness = self._ensure_tensor(
            kwargs.get("layer_thickness", self.capillary_profile_layer_thickness)
        ).to(device=water_table_depth.device, dtype=water_table_depth.dtype)
        soil_depth = layer_thickness * soil_layers

        # Compute capillary rise flux using Gardner approximation (tensor)
        surface_flux, flux_residuals = self._compute_gardner_flux_with_residuals(
            water_table_depth
        )

        # Compute flux profile with depth
        if self.enable_depth_profile:
            layer_depths = torch.linspace(
                layer_thickness / 2.0,
                soil_depth - layer_thickness / 2.0,
                soil_layers,
                device=water_table_depth.device,
                dtype=water_table_depth.dtype,
            )
            flux_profile = self._compute_flux_profile(water_table_depth, layer_depths)
        else:
            # Uniform flux approximation with broadcasting
            ones_shape = (soil_layers,) + tuple(surface_flux.shape)
            flux_profile = torch.ones(
                ones_shape, device=surface_flux.device, dtype=surface_flux.dtype
            ) * surface_flux

        # Apply saturation constraints if soil moisture provided
        if "soil_moisture_profile" in state:
            soil_moisture = self._ensure_tensor(state["soil_moisture_profile"]).to(
                device=flux_profile.device, dtype=flux_profile.dtype
            )
            unconstrained_profile = flux_profile
            flux_profile = self._apply_saturation_constraints(flux_profile, soil_moisture)
            saturation_residual = torch.clamp(
                unconstrained_profile - flux_profile, min=0.0
            )
        else:
            saturation_residual = torch.zeros_like(flux_profile)

        # Convert to mass flux units [kg m-2 s-1]
        water_density = self._ensure_tensor(self.water_density).to(
            device=surface_flux.device, dtype=surface_flux.dtype
        )
        surface_flux_mass = surface_flux * water_density
        flux_profile_mass = flux_profile * water_density

        # Compute diagnostic variables
        effective_height = torch.minimum(water_table_depth, max_height)
        # Water availability factor (0 = no capillary rise, 1 = maximum)
        availability_factor = torch.clamp(
            1.0 - water_table_depth / max_height, min=0.0, max=1.0
        )
        
        # The two limiter residuals have one output path each, the declared flat
        # diagnostics.capillary_rise.* UpdateItems below [kg m-2 s-1].
        conductivity_residual_mass = (
            flux_residuals["conductivity_clipping_residual"] * water_density
        )
        saturation_residual_mass = saturation_residual * water_density
        gardner_a, gardner_b, gardner_B = self._compute_gardner_coefficients(
            water_table_depth
        )
        diagnostics = {
            "effective_capillary_height": effective_height,              # m
            "capillary_rise_profile": flux_profile_mass,  # kg m-2 s-1
            "capillary_availability": availability_factor,               # dimensionless
            "gardner_a_coefficient": gardner_a,                          # m^(b+1) s-1
            "gardner_b_coefficient": gardner_b,                          # dimensionless
            "gardner_B_coefficient": gardner_B,                          # dimensionless
            "negative_flux_projection_residual": (
                flux_residuals["negative_flux_projection_residual"] * water_density
            ),
        }

        return {
            "capillary_rise_flux": UpdateItem(
                value=surface_flux_mass,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.capillary_rise": diagnostics,
            "diagnostics.capillary_rise.conductivity_clipping_residual": UpdateItem(
                value=conductivity_residual_mass,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.capillary_rise.saturation_constraint_residual": UpdateItem(
                value=saturation_residual_mass,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the open capillary-port limiter ledger.

        References:
          - Gardner (1958), steady capillary rise from a water table.
          - Eagleson (1978), reduced-order vertical soil-water movement.
        Mathematical expression:
          F_candidate = F_capillary + R_K - R_negative.
        Parameter meanings:
          F_capillary is the retained upward boundary mass flux, R_K is the
          conductivity-cap residual, and R_negative is the positivity
          projection residual; all terms use kg m-2 s-1.
        Open-system route:
          The reported flux leaves groundwater and enters the selected
          Richards soil boundary; storage closure is checked by those owners.
        """

        state = self._validate_state_inputs(input_state)
        water_table_depth = state["water_table_depth"].to(
            device=getattr(self, "device", torch.device("cpu"))
        )
        if not bool(torch.all(torch.isfinite(water_table_depth))):
            raise ValueError("Water table depth must contain only finite values")
        if bool(torch.any(water_table_depth < 0)):
            raise ValueError("Water table depth must be non-negative")

        reference_flux, reference_residuals = (
            self._compute_gardner_flux_with_residuals(water_table_depth)
        )
        water_density = self._ensure_tensor(self.water_density).to(
            device=reference_flux.device,
            dtype=reference_flux.dtype,
        )
        reference_candidate = (
            reference_flux
            + reference_residuals["conductivity_clipping_residual"]
            - reference_residuals["negative_flux_projection_residual"]
        ) * water_density

        def _require_tensor(name: str, value: Any) -> Tensor:
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(f"Conservation output '{name}' must be a torch.Tensor")
            if value.numel() == 0:
                raise ValueError(f"Conservation output '{name}' must be non-empty")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"Conservation output '{name}' must be finite")
            return value

        flux_key = "capillary_rise_flux"
        conductivity_key = (
            "diagnostics.capillary_rise.conductivity_clipping_residual"
        )
        for key in (flux_key, conductivity_key):
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")

        diagnostics_key = "diagnostics.capillary_rise"
        if diagnostics_key not in output_updates:
            raise KeyError(f"Missing conservation output '{diagnostics_key}'")
        diagnostics = output_updates[diagnostics_key]
        if isinstance(diagnostics, UpdateItem):
            diagnostics = diagnostics.value
        if not isinstance(diagnostics, Mapping):
            raise TypeError("Capillary-rise conservation diagnostics must be a mapping")
        negative_key = "negative_flux_projection_residual"
        if negative_key not in diagnostics:
            raise KeyError(f"Missing conservation diagnostic '{negative_key}'")

        reported_flux = _require_tensor(flux_key, output_updates[flux_key])
        reported_conductivity = _require_tensor(
            conductivity_key,
            output_updates[conductivity_key],
        )
        reported_negative = _require_tensor(
            negative_key,
            diagnostics[negative_key],
        )
        for name, value in (
            (flux_key, reported_flux),
            (conductivity_key, reported_conductivity),
            (negative_key, reported_negative),
        ):
            if value.shape != reference_candidate.shape:
                raise ValueError(
                    f"Conservation output '{name}' shape must match "
                    "water_table_depth"
                )

        reported_candidate = (
            reported_flux + reported_conductivity - reported_negative
        )
        magnitude = torch.max(torch.abs(reference_candidate - reported_candidate))
        return {"mass": float(magnitude.detach().item())}

    # References: Gardner (1958).
    # Formula: w = B * a / (Z^b).
    # Parameters: Z (groundwater depth, m), a and b (hydraulic coefficients), B (dimensionless).
    def _compute_gardner_flux_with_residuals(self, Z: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute Gardner flux and explicit projection residuals.

        References:
            - Gardner, W.R. (1958). Soil Science, 85(4), 228-232.

        Mathematical expression:
            w_raw = B * a / (Z ** b);
            w = I(0 < Z <= Zmax) * max(0, min(w_raw, Ks)).

        Parameter meanings:
            - Z: groundwater depth [m], positive downward
            - Ks: saturated hydraulic conductivity [m s-1]
            - B, a, b: Gardner capillary coefficients
        """
        # References:
        #   Gardner (1958); Eagleson (1978); retained Brooks-Corey/van Genuchten approximation.
        # Mathematical expression:
        #   Brooks-Corey: b=2+3*lambda_p, a=Ks*abs(h_entry)^b;
        #   van Genuchten approximation: b=2/m, a=Ks*(1/alpha)^b;
        #   B=1+1.5/(b-1), defined for b > 1; q=B*a/Z^b.
        # Parameter meanings:
        #   lambda_p is the pore-size index; alpha/m are van Genuchten controls;
        #   h_entry is air-entry head. All closures use the same limited flux owner.
        # Units:
        #   h_entry/Z [m], alpha [m-1], Ks/q [m s-1], b/B/m/lambda_p [1];
        #   a [m^(b+1) s-1].
        # Domain/constraints:
        #   The van Genuchten relation here is the documented reduced-order
        #   Gardner approximation, not an exact Richards solution. Invalid distance
        #   support returns zero and conductivity/projection residuals are explicit.
        # Sign convention:
        #   q is an upward nonnegative magnitude on valid unsaturated support.

        device = Z.device
        dtype = Z.dtype

        max_height = self._ensure_tensor(self.max_capillary_height).to(device=device, dtype=dtype)
        # The coefficients are derived here from the current held values (one
        # derivation per evaluation), then read on the device and dtype of Z.
        gardner_a, gardner_b, gardner_B = self._compute_gardner_coefficients(Z)
        gardner_B = self._ensure_tensor(gardner_B).to(device=device, dtype=dtype)
        gardner_a = self._ensure_tensor(gardner_a).to(device=device, dtype=dtype)
        gardner_b = self._ensure_tensor(gardner_b).to(device=device, dtype=dtype)
        k_saturated = self._ensure_tensor(self.k_saturated).to(device=device, dtype=dtype)

        valid = (Z > 0) & (Z <= max_height)
        safe_Z = torch.where(valid, Z, torch.ones_like(Z))
        raw_flux = gardner_B * gardner_a / torch.pow(safe_Z, gardner_b)
        conductivity_limited_flux = torch.minimum(raw_flux, k_saturated)
        positive_flux = torch.clamp(conductivity_limited_flux, min=0.0)
        flux = torch.where(valid, positive_flux, torch.zeros_like(Z))
        zero_residual = torch.zeros_like(Z)
        conductivity_clipping_residual = torch.where(
            valid,
            torch.clamp(raw_flux - k_saturated, min=0.0),
            zero_residual,
        )
        negative_flux_projection_residual = torch.where(
            valid,
            torch.clamp(-conductivity_limited_flux, min=0.0),
            zero_residual,
        )
        return flux, {
            "conductivity_clipping_residual": conductivity_clipping_residual,
            "negative_flux_projection_residual": negative_flux_projection_residual,
        }

    def _compute_gardner_flux_tensor(self, Z: Tensor) -> Tensor:
        """
        Compute capillary rise flux using Gardner approximation (tensor version).

        References:
            - Gardner, W.R. (1958). Soil Science, 85(4), 228-232.

        Mathematical expression:
            w = B * a / (Z ** b)

        Parameter meanings:
            - Z: groundwater depth [m], positive downward
            - a, b: hydraulic parameters in K(psi) = a * psi^b
            - B: Gardner coefficient (dimensionless)
        """
        # References:
        #   Gardner (1958); Eagleson (1978); retained Brooks-Corey/van Genuchten approximation.
        # Mathematical expression:
        #   Brooks-Corey: b=2+3*lambda_p, a=Ks*abs(h_entry)^b;
        #   van Genuchten approximation: b=2/m, a=Ks*(1/alpha)^b;
        #   B=1+1.5/(b-1), defined for b > 1; q=B*a/Z^b.
        # Parameter meanings:
        #   lambda_p is the pore-size index; alpha/m are van Genuchten controls;
        #   h_entry is air-entry head. All closures use the same limited flux owner.
        # Units:
        #   h_entry/Z [m], alpha [m-1], Ks/q [m s-1], b/B/m/lambda_p [1];
        #   a [m^(b+1) s-1].
        # Domain/constraints:
        #   The van Genuchten relation here is the documented reduced-order
        #   Gardner approximation, not an exact Richards solution. Invalid distance
        #   support returns zero and conductivity/projection residuals are explicit.
        # Sign convention:
        #   q is an upward nonnegative magnitude on valid unsaturated support.
        flux, _ = self._compute_gardner_flux_with_residuals(Z)
        return flux
    
    # References: Gardner (1958).
    # Formula: w = B * a / (Z^b).
    # Parameters: Z (water table depth, m), a and b (hydraulic coefficients), B (dimensionless).
    def _compute_gardner_flux(self, Z: float) -> float:
        """
        Compute capillary rise flux using Gardner approximation.
        
        Parameters
        ----------
        Z : float
            Water table depth [m]
            
        Returns
        -------
        float
            Capillary rise flux [m s-1]
            
        Mathematical Expression
        -----------------------
        Gardner approximation: w = B*a/(Z^b)
        where:
        - B = 1 + 1.5/(b-1)
        - a, b from hydraulic function K(psi) = a * psi^b
        """
        # References:
        #   Gardner (1958); Eagleson (1978); retained Brooks-Corey/van Genuchten approximation.
        # Mathematical expression:
        #   Brooks-Corey: b=2+3*lambda_p, a=Ks*abs(h_entry)^b;
        #   van Genuchten approximation: b=2/m, a=Ks*(1/alpha)^b;
        #   B=1+1.5/(b-1), defined for b > 1; q=B*a/Z^b.
        # Parameter meanings:
        #   lambda_p is the pore-size index; alpha/m are van Genuchten controls;
        #   h_entry is air-entry head. All closures use the same limited flux owner.
        # Units:
        #   h_entry/Z [m], alpha [m-1], Ks/q [m s-1], b/B/m/lambda_p [1];
        #   a [m^(b+1) s-1].
        # Domain/constraints:
        #   The van Genuchten relation here is the documented reduced-order
        #   Gardner approximation, not an exact Richards solution. Invalid distance
        #   support returns zero and conductivity/projection residuals are explicit.
        # Sign convention:
        #   q is an upward nonnegative magnitude on valid unsaturated support.
        # interface_continuity wrapper: keep float signature while using tensor path.
        flux_tensor = self._compute_gardner_flux_tensor(self._ensure_tensor(Z))
        flux_scalar = flux_tensor.detach().reshape(-1)[0].tolist()
        return float(flux_scalar)
    
    # References: Gardner (1958); Eagleson (1978).
    # Formula: w(z) = (B * a / (Z^b)) * exp(-z * porosity_decay).
    # Parameters: Z (water table depth, m), z (layer depth, m), porosity_decay (m-1), a, b, B.
    def _compute_flux_profile(self, gw_depth: Tensor, layer_depths: Tensor) -> Tensor:
        """
        Compute capillary flux profile with depth considering porosity variation.

        Parameters
        ----------
        gw_depth : Tensor
            Water table depth [m]
        layer_depths : Tensor
            Depths of soil layers [m]

        Returns
        -------
        Tensor
            Flux at each layer [m s-1]
        """
        # References:
        #   Gardner (1958); Eagleson (1978).
        # Mathematical expression:
        #   q(z)=q_Gardner(Z-z)*exp(-z/p) for z<Z; otherwise 0;
        #   q_limited=q(z)*clip((c_sat*n0-theta)/(c_band*n0),0,1).
        # Parameter meanings:
        #   z is layer-center depth; Z is groundwater depth; p is the positive
        #   porosity decay length, n0 surface porosity, c_sat/c_band configured factors.
        # Units:
        #   z/Z/p [m], theta/n0/c_sat/c_band [1], q [m s-1].
        # Domain/constraints:
        #   This profile is diagnostic. The forward owner only applies saturation
        #   limits for its currently accepted profile shape; it does not independently
        #   advance soil or groundwater or replace the surface flux with a profile sum.
        # Sign convention:
        #   Positive profile flow is upward; the saturation factor only reduces magnitude.
        gw_depth_t = self._ensure_tensor(gw_depth).to(
            device=layer_depths.device, dtype=layer_depths.dtype
        )
        layer_depths_t = self._ensure_tensor(layer_depths).to(
            device=gw_depth_t.device, dtype=gw_depth_t.dtype
        )

        depth_grid = layer_depths_t.view(-1, *([1] * gw_depth_t.ndim))
        gw_grid = gw_depth_t.unsqueeze(0)
        distance_to_wt = gw_grid - depth_grid

        porosity_decay = self._ensure_tensor(self.porosity_decay).to(
            device=gw_depth_t.device, dtype=gw_depth_t.dtype
        )
        porosity_factor = torch.exp(-depth_grid * porosity_decay)

        base_flux = self._compute_gardner_flux_tensor(distance_to_wt)
        valid = depth_grid < gw_grid
        flux_profile = torch.where(valid, base_flux * porosity_factor, torch.zeros_like(base_flux))

        return flux_profile
    
    def _apply_saturation_constraints(self, flux_profile: Tensor, soil_moisture: Tensor) -> Tensor:
        """
        Apply saturation constraints to limit capillary rise when soil is saturated.
        
        Parameters
        ----------
        flux_profile : Tensor
            Unconstrained flux profile [m s-1]
        soil_moisture : Tensor
            Current soil moisture profile [m3 m-3]
            
        Returns
        -------
        Tensor
            Constrained flux profile [m s-1]
        """
        # References:
        #   Gardner (1958); Eagleson (1978).
        # Mathematical expression:
        #   q(z)=q_Gardner(Z-z)*exp(-z/p) for z<Z; otherwise 0;
        #   q_limited=q(z)*clip((c_sat*n0-theta)/(c_band*n0),0,1).
        # Parameter meanings:
        #   z is layer-center depth; Z is groundwater depth; p is the positive
        #   porosity decay length, n0 surface porosity, c_sat/c_band configured factors.
        # Units:
        #   z/Z/p [m], theta/n0/c_sat/c_band [1], q [m s-1].
        # Domain/constraints:
        #   This profile is diagnostic. The forward owner only applies saturation
        #   limits for its currently accepted profile shape; it does not independently
        #   advance soil or groundwater or replace the surface flux with a profile sum.
        # Sign convention:
        #   Positive profile flow is upward; the saturation factor only reduces magnitude.
        surface_porosity = self._ensure_tensor(self.surface_porosity).to(
            device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        saturation_threshold_factor = self._ensure_tensor(
            self.capillary_saturation_threshold_factor
        ).to(device=soil_moisture.device, dtype=soil_moisture.dtype)
        saturation_bandwidth_factor = self._ensure_tensor(
            self.capillary_saturation_bandwidth_factor
        ).to(device=soil_moisture.device, dtype=soil_moisture.dtype)
        saturation_threshold = saturation_threshold_factor * surface_porosity
        
        # Reduce flux where soil is near saturation
        saturation_factor = torch.clamp(
            (saturation_threshold - soil_moisture)
            / (saturation_bandwidth_factor * surface_porosity),
            min=0.0, max=1.0
        )
        
        constrained_flux = flux_profile * saturation_factor
        
        return constrained_flux
    
    def _zero_flux_result(self, water_table_depth: Union[float, Tensor]) -> FluxDict:
        """Return zero flux result when capillary rise is not active."""
        depth_tensor = self._ensure_tensor(water_table_depth)
        layer_count = int(self.capillary_profile_layer_count)
        zeros_shape = (layer_count,) + tuple(depth_tensor.shape)
        zeros_profile = torch.zeros(zeros_shape, dtype=depth_tensor.dtype, device=depth_tensor.device)
        gardner_a, gardner_b, gardner_B = self._compute_gardner_coefficients(
            depth_tensor
        )
        diagnostics = {
            "effective_capillary_height": depth_tensor.new_tensor(0.0),
            "capillary_rise_profile": zeros_profile,
            "capillary_availability": depth_tensor.new_tensor(0.0),
            "gardner_a_coefficient": gardner_a,
            "gardner_b_coefficient": gardner_b,
            "gardner_B_coefficient": gardner_B,
            "negative_flux_projection_residual": depth_tensor.new_zeros(depth_tensor.shape),
        }
        return {
            "capillary_rise_flux": UpdateItem(
                value=depth_tensor.new_tensor(0.0),
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.capillary_rise": diagnostics,
            "diagnostics.capillary_rise.conductivity_clipping_residual": UpdateItem(
                value=depth_tensor.new_zeros(depth_tensor.shape),
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.capillary_rise.saturation_constraint_residual": UpdateItem(
                value=zeros_profile,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
        }

    def _validate_state_inputs(self, state: StateDict) -> StateDict:
        """
        Validate and normalize the canonical capillary-rise inputs.

        Parameters
        ----------
        state : StateDict
            Incoming state mapping.
        Returns
        -------
        StateDict
            Copy with canonical inputs coerced to tensors.
        """
        normalized: StateDict = dict(state or {})
        if normalized.get("water_table_depth") is None:
            raise ValueError("Missing required input: water_table_depth")
        normalized["water_table_depth"] = self._ensure_tensor(
            normalized["water_table_depth"]
        )
        if "soil_moisture_profile" in normalized and normalized["soil_moisture_profile"] is not None:
            soil_moisture = self._ensure_tensor(normalized["soil_moisture_profile"])
            if not torch.isfinite(soil_moisture).all():
                raise ValueError(
                    "soil_moisture_profile must contain only finite values"
                )
            normalized["soil_moisture_profile"] = soil_moisture
        return normalized

    def get_transport_metadata(self) -> Dict:
        """
        Get transport mechanism metadata for diagnostics.
        
        Returns
        -------
        Dict
            Transport mechanism metadata
        """
        gardner_a, gardner_b, gardner_B = self._compute_gardner_coefficients()
        return {
            "transport_type": self.transport_type,
            "max_capillary_height": self.max_capillary_height,
            "pore_size_index": self.pore_size_index,
            "air_entry_pressure": self.air_entry_pressure,
            "hydraulic_method": self.hydraulic_method,
            "gardner_a": gardner_a,
            "gardner_b": gardner_b,
            "gardner_B": gardner_B,
            "k_saturated": self.k_saturated
        }


# Export the main class
__all__ = ["CapillaryRiseTransport"]

