"""
Transport Operators for RELIC Unified Computation Layer

This module provides high-level transport operator implementations that wrap
the original function-based operators (apply_diffusion, apply_advection,
advance_advection_diffusion) with a class-based OperatorBase interface.

These are composite operators that combine multiple basic operators
(gradient, divergence, laplacian) to solve transport equations.

Mathematical Expressions:
    Diffusion: dc/dt = div(Kgradc)
    Advection: dc/dt = -div(v c)
    Advection-Diffusion: dc/dt = -div(v c) + div(Kgradc) + S

Design Pattern:
    These classes wrap the original apply_* and advance_* functions,
    preserving 100% scientific correctness while providing:
    - OperatorBase interface compliance
    - Factory instantiation support
    - ParameterManager integration
    - Comprehensive diagnostics

References:
    - Crank, J. (1975). The Mathematics of Diffusion. Oxford
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs. SIAM
    - Hundsdorfer & Verwer (2003). Time-Dependent Advection-Diffusion-Reaction Equations
"""

from typing import Any, Optional, Union, Tuple, Dict
import torch

from .base import (
    OperatorBase,
    OperatorMetadata,
    SpatialRequirements,
    reject_unexpected_runtime_parameters,
)
from .registry import register_operator

# Import original implementation functions
from .functional import (
    _check_advection_time_integration_pair,
    _check_splitting_time_integration_pair,
    apply_diffusion,
    apply_advection,
    advance_advection_diffusion
)

Tensor = torch.Tensor


def _align_tensor_to_state(value: Tensor, state: Tensor) -> Tensor:
    """Align one floating tensor to the runtime state's device and dtype.

    References:
      - RELIC device policy: the simulator registration owns the runtime device.
      - PyTorch type promotion rules for mixed-precision arithmetic.
    Mathematical expression:
      value' = value viewed in dtype(state) on device(state); state is untouched.
    Parameter meanings:
      value is a coefficient or velocity field; state is the runtime concentration.
    Units:
      unchanged by the cast.
    Domain/constraints:
      only real floating tensors are cast; complex, boolean and integer tensors
      pass through unchanged so the kernels reject them explicitly instead of
      silently discarding an imaginary part or promoting the state dtype.
    Sign convention:
      none.
    """
    if value.is_floating_point() and not value.is_complex():
        return value.to(device=state.device, dtype=state.dtype)
    return value


def _align_velocity_to_state(
    velocity: Union[Tensor, Tuple[Tensor, ...], Any],
    state: Tensor,
) -> Union[Tensor, Tuple[Tensor, ...], Any]:
    """Align tensor velocity components to the state without casting the state."""
    if isinstance(velocity, (tuple, list)):
        return tuple(
            _align_tensor_to_state(component, state)
            if isinstance(component, torch.Tensor)
            else component
            for component in velocity
        )
    if isinstance(velocity, torch.Tensor):
        return _align_tensor_to_state(velocity, state)
    return velocity


@register_operator("transport.diffusion", category="transport")
class DiffusionOperator(OperatorBase):
    """
    Diffusion operator: dc/dt = div(Kgradc)
    
    This class wraps the original apply_diffusion function,
    providing a high-level interface for diffusion processes.
    
    Mathematical Expression:
        dc/dt = div(Kgradc) = Klaplacianc  (for constant K)
        Discretized: c^(n+1) = c^n + dt*Klaplacianc^n
    
    Parameters (from ParameterManager or config):
        diffusivity : float
            Diffusion coefficient [m2 s-1]
        spatial_dims : int
            Number of spatial dimensions (1, 2, or 3)
        boundary_method : str
            Boundary treatment: "zero", "extrapolate", "periodic"
        check_stability : bool
            Check diffusion stability criterion
        enforce_non_negative : bool
            Enforce non-negativity constraint
    
    Usage Example:
        >>> from relic.computation.operators import OperatorFactory
        >>> op = OperatorFactory.create(
        ...     "transport.diffusion",
        ...     config={
        ...         "diffusivity": 1e-5,
        ...         "spatial_dims": 2,
        ...         "boundary_method": "periodic"
        ...     }
        ... )
        >>> result = op.apply(
        ...     concentration=c,
        ...     grid_spacing=(100.0, 100.0),
        ...     time_step=1800.0
        ... )
    
    References:
        - Crank, J. (1975). The Mathematics of Diffusion. Oxford
    """
    
    operator_type = "transport"
    
    def __init__(
        self,
        name: str = "diffusion_operator",
        config: Optional[dict] = None,
        metadata: Optional['OperatorMetadata'] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize diffusion operator."""
        # namespace and validate_parameters are owned by OperatorBase; they are
        # passed to it (assigning them before super() was overwritten there).
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=dtype,
            namespace=namespace or "computation.operators.transport",
            validate_parameters=validate_parameters,
        )
        
        # Get parameters
        self.diffusivity = self.get_parameter(
            'diffusivity',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.spatial_dims = self.get_parameter(
            'spatial_dims',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.boundary_method = self.get_parameter(
            'boundary_method',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.check_stability = self.get_parameter(
            'check_stability',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.enforce_non_negative = self.get_parameter(
            'enforce_non_negative',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        # Create metadata
        self.metadata = OperatorMetadata(
            name=f"{self.name}",
            operator_type="transport",
            parameters={
                "diffusivity": f"{self.diffusivity} m2 s-1",
                "spatial_dims": self.spatial_dims,
                "boundary_method": self.boundary_method
            },
            mathematical_expression="dc/dt = div(Kgradc)",
            description="Diffusion operator with stability checking",
            input_units="kg m-3",
            output_units="kg m-3 (time_step given) or kg m-3 s-1 (rate)",
            spatial_dimensions=0,  # spatial_dims is a runtime-overridable parameter
            references=[
                "Crank, J. (1975). The Mathematics of Diffusion. Oxford"
            ],
            spatial_requirements=SpatialRequirements(
                stencil_radius=1,
                halo_depth=1,
                locality="compact",
                min_resolution=None,
                preferred_layout="soa",
            ),
        )
    
    def apply(
        self,
        concentration: Tensor,
        grid_spacing: Union[float, Tuple[float, ...]],
        time_step: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Apply diffusion operator to concentration field.
        
        Parameters
        ----------
        concentration : Tensor
            Concentration field [kg m-3]
            Shape: (batch, *spatial_shape)
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing in each direction [m]
        time_step : Optional[float]
            Time step for explicit integration [s]
            If None, returns diffusion rate instead
        **kwargs
            Optional runtime overrides of the constructor parameters and
            the kernel controls; any other keyword raises TypeError
        
        Returns
        -------
        Dict[str, Tensor]
            {
                'concentration' or 'diffusion_rate': Updated field or rate,
                'stability_number': Diffusion stability number,
                'conservation_error': Relative conservation error (if time_step provided)
            }
        """
        # Runtime overrides replace the configured values for this call; the
        # kernel validates them. An unknown keyword is rejected, never ignored.
        reject_unexpected_runtime_parameters(
            kwargs,
            {
                "diffusivity", "spatial_dims", "boundary_method",
                "check_stability", "enforce_non_negative",
                "run_conservation_check", "time_integration_method",
                "implicit_midpoint_max_iter", "implicit_midpoint_tolerance",
                "boundary_state", "source_term", "diffusivity_layout",
            },
        )
        diffusivity = kwargs.get('diffusivity', self.diffusivity)
        spatial_dims = kwargs.get('spatial_dims', self.spatial_dims)
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        check_stability = kwargs.get('check_stability', self.check_stability)
        enforce_non_negative = kwargs.get('enforce_non_negative', self.enforce_non_negative)
        
        # Runtime state owns device, dtype, and autograd history. The operator
        # facade is stateless and must not silently migrate or narrow it.
        if isinstance(diffusivity, torch.Tensor):
            diffusivity_value = diffusivity.to(
                device=concentration.device,
                dtype=concentration.dtype,
            )
        else:
            diffusivity_value = float(diffusivity)

        # Call original implementation
        result = apply_diffusion(
            concentration=concentration,
            diffusivity=diffusivity_value,
            grid_spacing=grid_spacing,
            spatial_dims=int(spatial_dims),
            boundary_method=str(boundary_method),
            time_step=time_step,
            check_stability=bool(check_stability),
            enforce_non_negative=bool(enforce_non_negative),
            run_conservation_check=bool(
                kwargs.get('run_conservation_check', True)
            ),
            time_integration_method=kwargs.get('time_integration_method'),
            implicit_midpoint_max_iter=kwargs.get('implicit_midpoint_max_iter'),
            implicit_midpoint_tolerance=kwargs.get('implicit_midpoint_tolerance'),
            boundary_state=kwargs.get('boundary_state'),
            source_term=kwargs.get('source_term'),
            diffusivity_layout=kwargs.get('diffusivity_layout'),
        )
        
        return result


@register_operator("transport.advection", category="transport")
class AdvectionOperator(OperatorBase):
    """
    Advection operator: dc/dt = -div(v c)
    
    This class wraps the original apply_advection function,
    providing a high-level interface for advection processes.
    
    Mathematical Expression:
        dc/dt = -div(v c) = -v*gradc  (for incompressible flow)
        Upwind scheme for numerical stability
    
    Parameters (from ParameterManager or config):
        spatial_dims : int
            Number of spatial dimensions (1 or 2)
        scheme : str
            Advection scheme; one of the allowed_values of
            computation.operators.transport.scheme
        boundary_method : str
            Boundary treatment
        check_cfl : bool
            Check CFL stability condition
        time_integration_method : str
            Time integration method of every step, from
            transport.numerical.time_integration_method (config key
            time_integration_method). The (scheme, method) pair is checked
            at construction: central needs an implicit method and the fully
            discrete schemes need euler.

    References:
        - LeVeque, R.J. (2007). Finite Difference Methods. SIAM
    """
    
    operator_type = "transport"
    
    def __init__(
        self,
        name: str = "advection_operator",
        config: Optional[dict] = None,
        metadata: Optional['OperatorMetadata'] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize advection operator."""
        # namespace and validate_parameters are owned by OperatorBase; they are
        # passed to it (assigning them before super() was overwritten there).
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=dtype,
            namespace=namespace or "computation.operators.transport",
            validate_parameters=validate_parameters,
        )
        
        # Get parameters
        self.spatial_dims = self.get_parameter(
            'spatial_dims',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.scheme = self.get_parameter(
            'scheme',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        # The pure advection operator keeps its own boundary treatment
        # (outflow extrapolation) in <namespace>.advection.boundary_method;
        # <namespace>.boundary_method ('zero') serves the diffusion and
        # advection-diffusion operators.
        self.boundary_method = self.get_parameter(
            'boundary_method',
            namespace=f"{self.namespace}.advection",
            validate=self.validate_parameters
        )
        
        self.check_cfl = self.get_parameter(
            'check_cfl',
            namespace=self.namespace,
            validate=self.validate_parameters
        )

        # The step integrator is the transport time integration choice
        # (transport.numerical.time_integration_method, overridable by the
        # config key); a (scheme, method) pair that cannot step together is
        # rejected here, before any rate is returned.
        self.time_integration_method = self.get_parameter(
            'time_integration_method',
            namespace="transport.numerical",
            validate=self.validate_parameters
        )
        _check_advection_time_integration_pair(
            self.scheme, self.time_integration_method
        )

        # Create metadata
        self.metadata = OperatorMetadata(
            name=f"{self.name}",
            operator_type="transport",
            parameters={
                "spatial_dims": self.spatial_dims,
                "scheme": self.scheme,
                "boundary_method": self.boundary_method,
                "time_integration_method": self.time_integration_method,
            },
            mathematical_expression="dc/dt = -div(v c)",
            description="Advection operator with CFL checking",
            input_units="kg m-3",
            output_units="kg m-3 (time_step given) or kg m-3 s-1 (rate)",
            spatial_dimensions=0,  # spatial_dims is a runtime-overridable parameter
            references=[
                "LeVeque, R.J. (2007). Finite Difference Methods"
            ],
            spatial_requirements=SpatialRequirements(
                stencil_radius=1,
                halo_depth=1,
                locality="compact",
                min_resolution=None,
                preferred_layout="soa",
            ),
        )
    
    def apply(
        self,
        concentration: Tensor,
        velocity: Union[Tensor, Tuple[Tensor, ...]],
        grid_spacing: Union[float, Tuple[float, ...]],
        time_step: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Apply advection operator to concentration field.
        
        Parameters
        ----------
        concentration : Tensor
            Concentration field [kg m-3]
        velocity : Union[Tensor, Tuple[Tensor, ...]]
            Velocity field(s) [m s-1]
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing in each direction [m]
        time_step : Optional[float]
            Time step for integration [s]
        **kwargs
            Optional runtime overrides of the constructor parameters and
            the kernel controls; any other keyword raises TypeError
        
        Returns
        -------
        Dict[str, Tensor]
            {
                'concentration' or 'advection_rate': Updated field or rate,
                'cfl_number': Maximum CFL number
            }
        """
        # Runtime overrides replace the configured values for this call; the
        # kernel validates them. An unknown keyword is rejected, never ignored.
        reject_unexpected_runtime_parameters(
            kwargs,
            {
                "spatial_dims", "scheme", "boundary_method", "check_cfl",
                "time_integration_method", "implicit_midpoint_max_iter",
                "implicit_midpoint_tolerance", "boundary_state",
            },
        )
        spatial_dims = kwargs.get('spatial_dims', self.spatial_dims)
        scheme = kwargs.get('scheme', self.scheme)
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        check_cfl = kwargs.get('check_cfl', self.check_cfl)
        
        # Runtime state owns device, dtype, and autograd history. The operator
        # facade is stateless and must not silently migrate or narrow it.
        # References: LeVeque (2002); RELIC device policy, registration-owned device.
        # Mathematical expression: F=u*c is evaluated in the dtype of c.
        # Parameter meanings: c is the runtime state, u the velocity components.
        # Units: unchanged; only the dtype and device of u are aligned to c.
        # Domain/constraints: real floating velocity tensors are aligned; complex
        #   or non-floating inputs pass through so the kernel rejects them.
        # Sign convention: none changed by the alignment.
        velocity_value = _align_velocity_to_state(velocity, concentration)
        
        # Call original implementation
        result = apply_advection(
            concentration=concentration,
            velocity=velocity_value,
            grid_spacing=grid_spacing,
            spatial_dims=int(spatial_dims),
            scheme=str(scheme),
            boundary_method=str(boundary_method),
            time_step=time_step,
            check_cfl=bool(check_cfl),
            time_integration_method=kwargs.get(
                'time_integration_method', self.time_integration_method
            ),
            implicit_midpoint_max_iter=kwargs.get('implicit_midpoint_max_iter'),
            implicit_midpoint_tolerance=kwargs.get('implicit_midpoint_tolerance'),
            boundary_state=kwargs.get("boundary_state"),
        )
        
        return result


@register_operator("transport.advection_diffusion", category="transport")
class AdvectionDiffusionOperator(OperatorBase):
    """
    Advection-diffusion operator: dc/dt = -div(v c) + div(Kgradc) + S
    
    This class wraps the original advance_advection_diffusion function,
    providing a unified interface for coupled advection-diffusion transport.
    
    Mathematical Expression:
        dc/dt = -div(v c) + div(Kgradc) + S
        
        Operator splitting methods supported:
        - Sequential: advection -> diffusion
        - Symmetric (Strang): diffusion(dt/2) -> advection(dt) -> diffusion(dt/2)
        - Combined: explicit coupled scheme
    
    Parameters (from ParameterManager or config):
        diffusivity : float
            Diffusion coefficient [m2 s-1]
        spatial_dims : int
            Number of spatial dimensions
        splitting_method : str
            "sequential", "symmetric", or "combined"
        advection_scheme : str
            Advection scheme
        boundary_method : str
            Boundary treatment
        check_stability : bool
            Check stability constraints
        enforce_non_negative : bool
            Enforce non-negativity
        time_integration_method : str
            Time integration method of every step, from
            transport.numerical.time_integration_method (config key
            time_integration_method). At construction the advection
            (scheme, method) pair and the (splitting, method) pair are
            checked: central needs an implicit method, the fully discrete
            schemes and combined splitting need euler.

    References:
        - Hundsdorfer & Verwer (2003). Time-Dependent ADR Equations
        - Strang, G. (1968). On construction of difference schemes. SIAM
    """
    
    operator_type = "transport"
    
    def __init__(
        self,
        name: str = "advection_diffusion_operator",
        config: Optional[dict] = None,
        metadata: Optional['OperatorMetadata'] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        namespace: Optional[str] = None,
        validate_parameters: bool = True
    ):
        """Initialize advection-diffusion operator."""
        # namespace and validate_parameters are owned by OperatorBase; they are
        # passed to it (assigning them before super() was overwritten there).
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
            dtype=dtype,
            namespace=namespace or "computation.operators.transport",
            validate_parameters=validate_parameters,
        )
        
        # Get parameters
        self.diffusivity = self.get_parameter(
            'diffusivity',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.spatial_dims = self.get_parameter(
            'spatial_dims',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.splitting_method = self.get_parameter(
            'splitting_method',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.advection_scheme = self.get_parameter(
            'advection_scheme',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.boundary_method = self.get_parameter(
            'boundary_method',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.check_stability = self.get_parameter(
            'check_stability',
            namespace=self.namespace,
            validate=self.validate_parameters
        )
        
        self.enforce_non_negative = self.get_parameter(
            'enforce_non_negative',
            namespace=self.namespace,
            validate=self.validate_parameters
        )

        # The step integrator is the transport time integration choice
        # (transport.numerical.time_integration_method, overridable by the
        # config key); pairs that cannot step together are rejected here.
        self.time_integration_method = self.get_parameter(
            'time_integration_method',
            namespace="transport.numerical",
            validate=self.validate_parameters
        )
        _check_advection_time_integration_pair(
            self.advection_scheme, self.time_integration_method
        )
        _check_splitting_time_integration_pair(
            self.splitting_method, self.time_integration_method
        )

        # Create metadata
        self.metadata = OperatorMetadata(
            name=f"{self.name}",
            operator_type="transport",
            parameters={
                "diffusivity": f"{self.diffusivity} m2 s-1",
                "spatial_dims": self.spatial_dims,
                "splitting_method": self.splitting_method,
                "boundary_method": self.boundary_method,
                "time_integration_method": self.time_integration_method,
            },
            mathematical_expression="dc/dt = -div(v c) + div(Kgradc) + S",
            description="Advection-diffusion operator with operator splitting",
            input_units="kg m-3",
            output_units="kg m-3",
            spatial_dimensions=0,  # spatial_dims is a runtime-overridable parameter
            references=[
                "Hundsdorfer & Verwer (2003). Time-Dependent ADR Equations",
                "Strang, G. (1968). On construction of difference schemes"
            ],
            spatial_requirements=SpatialRequirements(
                stencil_radius=1,
                halo_depth=1,
                locality="compact",
                min_resolution=None,
                preferred_layout="soa",
            ),
        )
    
    def apply(
        self,
        concentration: Tensor,
        velocity: Union[Tensor, Tuple[Tensor, ...]],
        grid_spacing: Union[float, Tuple[float, ...]],
        time_step: float,
        source_term: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply advection-diffusion operator to advance solution.
        
        Parameters
        ----------
        concentration : Tensor
            Concentration field [kg m-3]
        velocity : Union[Tensor, Tuple[Tensor, ...]]
            Velocity field(s) [m s-1]
        grid_spacing : Union[float, Tuple[float, ...]]
            Grid spacing in each direction [m]
        time_step : float
            Time step [s]
        source_term : Optional[Tensor]
            Source/sink term [kg m-3 s-1]
        **kwargs
            Optional runtime parameter overrides, including explicit advective
            and diffusive boundary states [kg m-3].
        
        Returns
        -------
        Dict[str, Tensor]
            {
                'concentration': Updated concentration [kg m-3],
                'advection_rate': Advection rate [kg m-3 s-1],
                'diffusion_rate': Diffusion rate [kg m-3 s-1],
                'cfl_number': Maximum CFL number,
                'stability_number': Diffusion stability number,
                'peclet_number': Maximum Peclet number,
                'conservation_error': Relative conservation error
            }
        """
        # Runtime overrides replace the configured values for this call; the
        # kernel validates them. An unknown keyword is rejected, never ignored.
        reject_unexpected_runtime_parameters(
            kwargs,
            {
                "diffusivity", "spatial_dims", "splitting_method",
                "advection_scheme", "boundary_method", "check_stability",
                "enforce_non_negative", "check_conservation",
                "time_integration_method", "implicit_midpoint_max_iter",
                "implicit_midpoint_tolerance", "advection_boundary_state",
                "diffusion_boundary_state", "diffusivity_layout",
            },
        )
        diffusivity = kwargs.get('diffusivity', self.diffusivity)
        spatial_dims = kwargs.get('spatial_dims', self.spatial_dims)
        splitting_method = kwargs.get('splitting_method', self.splitting_method)
        advection_scheme = kwargs.get('advection_scheme', self.advection_scheme)
        boundary_method = kwargs.get('boundary_method', self.boundary_method)
        check_stability = kwargs.get('check_stability', self.check_stability)
        enforce_non_negative = kwargs.get('enforce_non_negative', self.enforce_non_negative)
        check_conservation = kwargs.get('check_conservation', True)
        time_integration_method = kwargs.get(
            'time_integration_method', self.time_integration_method
        )
        implicit_midpoint_max_iter = kwargs.get('implicit_midpoint_max_iter')
        implicit_midpoint_tolerance = kwargs.get('implicit_midpoint_tolerance')
        advection_boundary_state = kwargs.get('advection_boundary_state')
        diffusion_boundary_state = kwargs.get('diffusion_boundary_state')
        
        # Runtime state owns device, dtype, and autograd history. The operator
        # facade is stateless and must not silently migrate or narrow it.
        # References: Hundsdorfer and Verwer (2003); RELIC device policy.
        # Mathematical expression: -div(u c)+div(K grad c)+S evaluated in dtype(c).
        # Parameter meanings: c runtime state, u velocity, K diffusivity (scalar
        #   or layer-wise tensor), S source handled by the canonical kernel.
        # Units: unchanged; only dtype/device of u and tensor K are aligned to c.
        # Domain/constraints: real floating tensors are aligned; complex or
        #   non-floating inputs pass through so the kernel rejects them.
        # Sign convention: none changed by the alignment.
        velocity_value = _align_velocity_to_state(velocity, concentration)
        if isinstance(diffusivity, torch.Tensor):
            diffusivity_value = _align_tensor_to_state(diffusivity, concentration)
        else:
            diffusivity_value = float(diffusivity)

        # Call original implementation
        result = advance_advection_diffusion(
            concentration=concentration,
            velocity=velocity_value,
            diffusivity=diffusivity_value,
            grid_spacing=grid_spacing,
            time_step=float(time_step),
            spatial_dims=int(spatial_dims),
            splitting_method=str(splitting_method),
            advection_scheme=str(advection_scheme),
            boundary_method=str(boundary_method),
            check_stability=bool(check_stability),
            enforce_non_negative=bool(enforce_non_negative),
            source_term=source_term,
            run_conservation_check=bool(check_conservation),
            time_integration_method=time_integration_method,
            implicit_midpoint_max_iter=implicit_midpoint_max_iter,
            implicit_midpoint_tolerance=implicit_midpoint_tolerance,
            advection_boundary_state=advection_boundary_state,
            diffusion_boundary_state=diffusion_boundary_state,
            diffusivity_layout=kwargs.get('diffusivity_layout'),
        )
        
        return result


# Export all classes
__all__ = [
    'DiffusionOperator',
    'AdvectionOperator',
    'AdvectionDiffusionOperator'
]
