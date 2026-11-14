"""
Functor Bridge Classes for RELIC Framework

This module implements the bridge classes that unify the FunctorBase execution
shell with the four abstract base classes (PhysicalField, BiologicalProcess,
TransportMechanism, ScaleCoupling). These bridge classes provide the standard
pattern for all executable scientific modules.

Bridge classes solve the inheritance pattern inconsistency by providing:
1. Single inheritance pattern for all executable functors
2. Unified parameter access through FunctorBase
3. Enhanced capability access through abstract base classes
4. Proper MRO (Method Resolution Order) handling
5. Automatic metadata and category assignment
6. Unit consistency validation and conversion

Design Pattern: Bridge Pattern + Multiple Inheritance
Mathematical Foundation: Category Theory Functors with Proper Composition

References:
    - Gang of Four Design Patterns - Bridge Pattern
    - Mac Lane, S. (1998). Categories for the Working Mathematician
    - Python Multiple Inheritance and MRO Documentation
    - CF-1.8 Climate and Forecast Metadata Conventions
"""

import torch
from typing import Dict, Any, ClassVar, Optional, Union, List, Tuple

from torch import Tensor
import logging

from .functor_base import FunctorBase
from .functor_types import ConfigDict, DependencyDeclaration, FunctorMetadata, ProcessCategory
from .functor_bridge_utils import BridgeParameterMixin
from ...contracts.contract_loader import load_contract_registry
from ..abstractions import (
    TransportMechanism,
    StateDict,
    FluxDict,
)

logger = logging.getLogger(__name__)

class TransportMechanismFunctor(BridgeParameterMixin, TransportMechanism, FunctorBase):  # type: ignore[misc,unused-ignore]
    """
    Bridge Class: Transport Mechanism Functor
    
    Executable functor for transport processes including diffusion,
    advection, turbulent mixing, and phase transfer mechanisms.
    
    Combines FunctorBase execution shell with TransportMechanism capabilities:
    - Spatial flux calculations between sources and sinks
    - Boundary condition enforcement
    - Diffusivity tensor computations
    - Conservation law validation
    - Multi-phase transport handling
    
    Usage Example:
        >>> class DiffusionTransport(TransportMechanismFunctor):
        ...     def forward(self, state, time, dt, **kwargs):
        ...         # Use TransportMechanism methods with forward state
        ...         source = state['source']
        ...         sink = state['sink']
        ...         flux = self.flux_to_updates({'diffusive_flux': source - sink})
        ...         # Use FunctorBase parameter access
        ...         diffusivity = FunctorBase.get_parameter_value(self, 'diffusivity')
        ...         return {'diffusive_flux': flux['diffusive_flux'] * diffusivity}

    A concrete class declares its descriptive label as the class attribute
    ``TRANSPORT_TYPE`` (one of ``TransportMechanism.TRANSPORT_TYPES``); a
    process whose law uses a constant D calls ``set_scalar_diffusivity`` after
    reading its parameters.
    """

    TRANSPORT_TYPE: ClassVar[Optional[str]] = None

    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return the instance-declared dependencies (bridge default).

        Returns:
            DependencyDeclaration: Names declared through config, set_dependencies
            and add_dependency. A subclass with a static declaration returns
            ``self._merge_declared_dependencies(static)`` instead.
        """
        return self._merge_declared_dependencies()
    
    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[ConfigDict] = None,
        metadata: Optional[FunctorMetadata] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize transport mechanism functor with cooperative inheritance.

        Parameters:
            name : str
                Functor name
            config : Optional[ConfigDict]
                Configuration dictionary
            metadata : Optional[FunctorMetadata]
                Process metadata
            **kwargs : Any
                Additional keyword arguments
        """
        # Handle name
        if name is None:
            name = kwargs.pop('name', 'transport_mechanism')

        # Handle config
        if config is None:
            config = kwargs.pop('config', {})

        # Ensure metadata category is correct
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                category=ProcessCategory.TRANSPORT_MECHANISM,
                description=f"Transport mechanism functor: {name}"
            )
        elif metadata.category is None:
            metadata.category = ProcessCategory.TRANSPORT_MECHANISM

        # The label is the declared class attribute, read before
        # TransportMechanism.__init__; the scalar D is never a constructor input.
        for keyword in ("transport_type", "diffusivity"):
            if keyword in kwargs:
                raise TypeError(
                    f"{type(self).__name__}: '{keyword}' is not a constructor keyword; "
                    "the label is the class attribute TRANSPORT_TYPE and a constant D is "
                    "set through set_scalar_diffusivity after the parameters are read"
                )
        transport_type = type(self).TRANSPORT_TYPE
        if transport_type is None:
            raise TypeError(
                f"{type(self).__name__} must declare the class attribute TRANSPORT_TYPE "
                f"(one of {sorted(TransportMechanism.TRANSPORT_TYPES)})"
            )
        peclet_number = kwargs.pop('peclet_number', None)
        device = kwargs.pop('device', None)
        resolved_device = (
            torch.device(device)
            if device is not None
            else (torch.device("cpu"))
        )

        # Store additional transport kwargs for later use
        self._transport_kwargs = {}
        for key in ['velocity_field', 'grid_spacing', 'spatial_dims', 'advection_scheme',
                    'diffusion_scheme', 'non_negative', 'splitting_method', 'time_integration']:
            if key in kwargs:
                self._transport_kwargs[key] = kwargs.pop(key)

        # Initialize TransportMechanism first (scientific rules layer)
        TransportMechanism.__init__(
            self,
            transport_type=transport_type,
            peclet_number=peclet_number,
            device=resolved_device
        )

        # Initialize FunctorBase (system interaction layer)
        FunctorBase.__init__(
            self,
            name=name,
            config=config,
            metadata=metadata,
            device=resolved_device,
            **kwargs,
        )

        # Initialize additional transport parameters
        self._init_transport_parameters()
        self._sync_transport_metadata()
        self._sync_conservation_targets_from_metadata()
    
    def _init_transport_parameters(self) -> None:
        """Attach stored transport kwargs to the instance.

        ``transport_type`` and the (initially absent) diffusivity come only
        from ``TransportMechanism.__init__``; the bridge fabricates neither.
        """
        missing = [name for name in ("transport_type", "diffusivity") if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__} has no {missing}: TransportMechanism.__init__ "
                "was bypassed, and the bridge does not fabricate transport attributes"
            )
        # Store any transport kwargs that were passed for child classes
        if hasattr(self, '_transport_kwargs'):
            for key, value in self._transport_kwargs.items():
                if not hasattr(self, key):
                    setattr(self, key, value)

    def _sync_transport_metadata(self) -> None:
        """Update FunctorMetadata with transport defaults."""
        metadata = getattr(self, 'metadata', None)
        if metadata is None:
            return

        if metadata.references is None:
            metadata.references = []
        transport_refs = [
            "Fick, A. (1855). On liquid diffusion.",
            "Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics."
        ]
        for ref in transport_refs:
            if ref not in metadata.references:
                metadata.references.append(ref)

        if not isinstance(metadata.parameters, dict):
            metadata.parameters = {}

        if metadata.conservation_laws:
            if isinstance(metadata.conservation_laws, list):
                metadata.conservation_laws = {
                    law: {'quantity': law} for law in metadata.conservation_laws
                }
        else:
            metadata.conservation_laws = {}

    @staticmethod
    def merge_source_sink(source: Any, sink: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Build a canonical state payload for transport forward() implementations.
        """
        state: Dict[str, Any] = {"source": source, "sink": sink}
        state.update(kwargs)
        return state

    def flux_to_updates(
        self,
        fluxes: Any,
        dt: Optional[Union[float, Tensor]] = None,
        flux_key: Optional[str] = None,
    ) -> FluxDict:
        """
        Convert flux(es) to state increments.

        References:
            - LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations.

        Mathematical Expression:
            - Increment: Delta = F * dt

        Parameters:
            fluxes : Any
                Either a FluxDict or a scalar/tensor flux.
            dt : Optional[float | Tensor]
                Time step in seconds, required when fluxes is a scalar/tensor.
            flux_key : Optional[str]
                Output key to use when fluxes is a scalar/tensor.

        Returns:
            FluxDict: Mapping from update key to increment tensor.
        """
        if isinstance(fluxes, dict) and dt is None and flux_key is None:
            return fluxes

        if dt is None or flux_key is None:
            raise TypeError("flux_to_updates expects (fluxes_dict) or (flux, dt, flux_key)")

        device = getattr(self, "device", torch.device("cpu"))
        flux_tensor = fluxes if isinstance(fluxes, Tensor) else torch.as_tensor(float(fluxes), device=device)
        dt_tensor = dt if isinstance(dt, Tensor) else torch.as_tensor(float(dt), device=device)
        return {flux_key: flux_tensor.to(device=device) * dt_tensor.to(device=device)}

    def validate_transport_units(self, fluxes_or_units: Any, element: Optional[str] = None) -> Optional[bool]:
        """
        Validate transport flux units for consistency.

        Parameters:
            fluxes_or_units : Any
                Either a FluxDict or a units string.
            element : Optional[str]
                Element/category name when validating a units string.

        Returns:
            Optional[bool]: When validating a units string, returns True/False; otherwise returns None.
        """
        if isinstance(fluxes_or_units, dict) and element is None:
            for key, value in fluxes_or_units.items():
                if not isinstance(value, Tensor):
                    logger.warning(f"Non-tensor flux in transport {self.name}: {key}")
            return None

        if isinstance(fluxes_or_units, str) and isinstance(element, str):
            units = fluxes_or_units.strip()
            has_time = ("s-1" in units) or ("s^-1" in units) or ("/s" in units)
            has_area = ("m-2" in units) or ("m^-2" in units) or ("/m2" in units)
            return bool(has_time and has_area)

        raise TypeError("validate_transport_units expects (fluxes_dict) or (units_str, element_str)")


    def finalize_transport_updates(self,
                                   outputs: FluxDict,
                                   dt: Union[float, Tensor],
                                   state: Optional[StateDict] = None,
                                   mapping: Optional[Dict[str, str]] = None,
                                   drop_diagnostics: bool = True) -> FluxDict:
        """
        Standardize transport outputs into state increments (deltas).

        The ``mapping`` declares every flux output and its target state
        variable; an output that is neither mapped nor in the ``diagnostics.``
        namespace raises. Bounds and budgets of the targets are not judged
        here: StateManager enforces declared bounds at commit and the
        documented conservation layers own the budgets.

        References:
            - Courant, R., Friedrichs, K., & Lewy, H. (1928). Uber die partiellen Differenzengleichungen.
            - Von Neumann, J., & Richtmyer, R. D. (1950). A method for the numerical calculation.
            - LeVeque, R. J. (2007). Finite Difference Methods for ODE/PDE.
            - Patankar, S. V. (1980). Numerical Heat Transfer and Fluid Flow.

        Mathematical expressions:
            - Flux to increment: Delta_y = F * Delta_t, with F converted from
              its declared unit to the per-second rate of the target contract
              unit (see _apply_flux_unit_conversion).

        Parameters:
            outputs : Dict[str, Tensor | float]
                Raw outputs from a transport functor (may contain fluxes and diagnostics).
            dt : float | Tensor
                Timestep in seconds.
            state : Optional[Dict]
                Current state carrying the declared flux units and the
                per-cell conversion context.
            mapping : Optional[Dict[str, str]]
                Declared mapping from flux output keys to target state
                variable names; the only way an output becomes a flux.
            drop_diagnostics : bool
                If True, outputs in the ``diagnostics.`` namespace are dropped;
                otherwise they are passed through unchanged.

        Raises:
            ValueError: an output key is neither mapped nor a ``diagnostics.``
                key, or a mapped flux cannot be converted (see
                _apply_flux_unit_conversion).
            TypeError: a mapped flux or a passed-through diagnostic is not numeric.

        Returns:
            Dict[str, Tensor]: Increments suitable for StateManager.apply_updates.
        """
        if outputs is None:
            return {}

        device = getattr(
            self,
            'device',
            torch.device("cpu"),
        )
        dt_tensor = self._to_transport_dt_tensor(dt, device)
        declared_targets = dict(mapping) if mapping else {}

        increments: FluxDict = {}
        fluxes_for_conservation: Dict[str, list[Tuple[Tensor, Tensor]]] = {}
        unit_conversions_performed: List[Dict[str, Any]] = []

        for key, val in outputs.items():
            target_key = declared_targets.get(key)
            if target_key is None:
                if not key.startswith(self._DIAGNOSTICS_PREFIX):
                    raise ValueError(
                        f"Transport output {key!r} of {self.name!r} has no declared "
                        "transport target: map it to its target state variable in "
                        "`mapping` or name it in the 'diagnostics.' namespace"
                    )
                if not drop_diagnostics:
                    increments[key] = self._require_transport_numeric_tensor(
                        key, val, device
                    )
                continue

            value_tensor = self._require_transport_numeric_tensor(key, val, device)
            converted_flux, conversion_info = self._apply_flux_unit_conversion(
                value_tensor,
                key,
                target_key,
                state,
            )
            if conversion_info.get('converted'):
                unit_conversions_performed.append(conversion_info)

            # References: LeVeque (2007), Finite Volume Methods, Section 4.1.
            # Mathematical expression: delta = F * dt, committed exactly.
            # Parameter meanings: F flux converted to the target rate unit;
            #   dt the transport step; delta the state increment.
            # Units: F [u_state s-1]; dt [s]; delta [u_state].
            # Domain/constraints: no magnitude filter: every increment reaches
            #   StateManager, whose declared bounds, clip_loss and conservation
            #   tolerances own roundoff; nothing is removed here.
            # Sign convention: delta keeps the sign of F (into the target > 0).
            delta = converted_flux.to(device) * dt_tensor.to(device)
            self._accumulate_transport_increment(
                target_key=target_key,
                converted_flux=converted_flux,
                delta=delta,
                increments=increments,
                fluxes_for_conservation=fluxes_for_conservation,
            )

        if self._should_track_conservation():
            conservation_diagnostics = self._compute_conservation_diagnostics(
                fluxes_for_conservation, state, dt_tensor
            )
            increments.update(conservation_diagnostics)

        if unit_conversions_performed:
            logger.debug(f"Unit conversions performed in {self.name}: {unit_conversions_performed}")

        return increments

    def _to_transport_dt_tensor(self, dt: Union[float, Tensor], device: torch.device) -> Tensor:
        """Normalize dt input to tensor on transport device."""
        if isinstance(dt, Tensor):
            return dt
        return torch.tensor(float(dt), device=device)

    # Outputs in this namespace are diagnostics, never state increments.
    _DIAGNOSTICS_PREFIX = "diagnostics."

    def _require_transport_numeric_tensor(
        self,
        key: str,
        value: Any,
        device: torch.device,
    ) -> Tensor:
        """Return a numeric transport output as a tensor; anything else raises."""
        if isinstance(value, Tensor):
            return value
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return torch.tensor(float(value), device=device)
        raise TypeError(
            f"Transport output {key!r} of {self.name!r} must be a tensor or a "
            f"number, got {type(value).__name__}"
        )

    def _accumulate_transport_increment(
        self,
        target_key: str,
        converted_flux: Tensor,
        delta: Tensor,
        increments: FluxDict,
        fluxes_for_conservation: Dict[str, List[Tuple[Tensor, Tensor]]],
    ) -> None:
        """Accumulate delta and conservation tuples for one resolved transport target."""
        if target_key not in fluxes_for_conservation:
            fluxes_for_conservation[target_key] = []
        fluxes_for_conservation[target_key].append((converted_flux, delta))
        if target_key in increments:
            increments[target_key] = increments[target_key] + delta
        else:
            increments[target_key] = delta

    def _should_track_conservation(self) -> bool:
        """
        Check if conservation tracking should be performed.
        
        Returns:
            bool: True if conservation tracking is enabled
        """
        try:
            track_flag = FunctorBase.get_parameter_value(
                self,
                name='track_conservation',
                namespace='transport.numerical',
                source_order=('manager',),
                validate=True,
                required=True,
            )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            raise RuntimeError(
                "Transport conservation flag 'transport.numerical.track_conservation' "
                "must be provided via ParameterManager."
            ) from exc
        return bool(track_flag)

    def _compute_conservation_diagnostics(
        self,
        fluxes_for_conservation: Dict[str, list[Tuple[Tensor, Tensor]]],
        state: Optional[StateDict],
        dt: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute conservation diagnostics for transport processes.
        
        Parameters:
            fluxes_for_conservation : Dict
                Fluxes grouped by target variable
            state : Optional[StateDict]
                Current state (may be None)
            dt : Tensor
                Time step
                
        Returns:
            Dict[str, Tensor]: Conservation diagnostics
        """
        # References: LeVeque (2007), Section 4.1 (flux-form increments).
        # Mathematical expression: total_flux_v = sum_i sum_cells F_i;
        #   relative_change_v = |sum_i sum_cells delta_i| / |sum_cells x_v| with
        #   delta_i = F_i * dt already integrated over the step.
        # Parameter meanings: F_i fluxes routed into target v; delta_i their
        #   increments; x_v the target state before the step.
        # Units: total_flux [u_state s-1]; relative_change dimensionless.
        # Domain/constraints: diagnostics only; the relative change is reported
        #   where sum(x_v) != 0; shape or device errors propagate (no logging
        #   fallback) because these diagnostics run on the governed path.
        # Sign convention: total_flux > 0 adds to the target.
        diagnostics: Dict[str, Tensor] = {}
        dt_tensor = dt.to(self.device) if isinstance(dt, Tensor) else torch.tensor(
            float(dt), device=self.device
        )
        for var_name, flux_list in fluxes_for_conservation.items():
            state_tensor = (
                state[var_name].to(dt_tensor.device)
                if state and var_name in state
                else None
            )
            total_flux = torch.zeros_like(state_tensor) if state_tensor is not None else None
            for flux, _ in flux_list:
                flux_tensor = flux.to(dt_tensor.device)
                if total_flux is None:
                    total_flux = torch.zeros_like(flux_tensor)
                total_flux = total_flux + flux_tensor

            diagnostics[f'diagnostics.transport.total_flux_{var_name}'] = torch.sum(total_flux)
            diagnostics[f'diagnostics.transport.num_fluxes_{var_name}'] = torch.tensor(
                len(flux_list), device=dt_tensor.device
            )

            if state_tensor is not None:
                current_mass = torch.sum(state_tensor)
                if bool(current_mass != 0):
                    total_change = torch.sum(
                        torch.stack([delta.to(dt_tensor.device) for _, delta in flux_list])
                    )
                    diagnostics[f'diagnostics.transport.relative_change_{var_name}'] = (
                        torch.abs(total_change) / torch.abs(current_mass)
                    )

        return diagnostics

    def _initialize_flux_conversion_info(
        self,
        flux_key: str,
        target_key: str,
    ) -> Dict[str, Any]:
        """Create default conversion metadata for flux conversions."""
        return {
            'flux_key': flux_key,
            'target_key': target_key,
            'converted': False,
            'conversion_type': None,
            'original_units': 'unknown',
            'target_units': 'unknown',
            'conversion_factor': 1.0
        }

    def _get_unit_conversion_inputs(
        self,
        state: Optional[StateDict],
        unit_converter: Any,
        flux_key: str,
        target_key: str,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Extract the base conversion context and the declared units map.

        References:
            - CF-1.8 Climate and Forecast Metadata Conventions
            - UDUNITS-2 Unit Conversion System

        Mathematical Expressions:
            Unit consistency rule: F_target = F_source * conversion_factor

        The base context carries the per-cell state fields the UnitConverter
        reads by their own names (water density, temperature, pressure). It
        carries no layer thickness: a thickness belongs to the variable being
        converted and is resolved from that variable's contract
        (``layer_thickness_variable``) only by a conversion that needs it.

        Parameters:
            state : Optional[StateDict]
                Current state dictionary with optional unit metadata.
            unit_converter : Any
                Unit converter instance used to build conversion context.
            flux_key : str
                Flux variable key.
            target_key : str
                Target state variable key.

        Returns:
            Tuple[Optional[Any], Optional[Dict[str, Any]]]
                (base conversion context, declared units map keyed by flux and
                target names)
        """
        context = None
        if state:
            context = unit_converter.get_conversion_context(state)

        units_map = None
        if state:
            if "units" in state and state["units"] is not None:
                candidate_units = state["units"]
            elif "output_units" in state and state["output_units"] is not None:
                candidate_units = state["output_units"]
            elif "flux_units" in state and state["flux_units"] is not None:
                candidate_units = state["flux_units"]
            else:
                candidate_units = None
            if isinstance(candidate_units, dict):
                units_map = candidate_units

        return context, units_map

    def _declared_layer_thickness_variable(self, variable_name: str) -> Optional[str]:
        """Return the layer-thickness state variable declared by a variable contract."""
        registry = load_contract_registry()
        contract = registry.get_contract(registry.resolve_alias(variable_name))
        if contract is None:
            return None
        return contract.layer_thickness_variable

    # The transport increment is delta = F * dt with dt in seconds, so a flux
    # is routed into its target in the per-second rate of the target's units.
    _PER_SECOND_SUFFIX = " s-1"

    def _declared_target_rate_units(
        self,
        target_key: str,
        units_map: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str]:
        """
        Return the canonical target name, its contract udunits and its rate unit.

        The target unit is owned by the target's variable contract
        (``dimension.udunits``); the flux is converted to the per-second rate of
        that unit because ``finalize_transport_updates`` applies
        ``delta = F * dt`` with ``dt`` in seconds. A unit that the state's units
        mapping declares for the target itself must equal the contract unit.

        Raises:
            ValueError: the target has no variable contract, the contract
                declares no udunits, or a declared target unit disagrees with it.
        """
        registry = load_contract_registry()
        canonical_target = registry.resolve_alias(target_key)
        contract = registry.get_contract(canonical_target)
        if contract is None:
            raise ValueError(
                f"Transport target {target_key!r} has no variable contract; the unit "
                "a flux is converted to comes only from the target's contract "
                "(dimension.udunits) under relic/contracts/variables"
            )
        state_units = str(contract.udunits or "").strip()
        if not state_units:
            raise ValueError(
                f"Variable contract {canonical_target!r} declares no udunits"
            )
        declared_target = units_map.get(target_key) if units_map else None
        if declared_target is not None and str(declared_target).strip() != state_units:
            raise ValueError(
                f"The declared unit {str(declared_target).strip()!r} of transport "
                f"target {target_key!r} conflicts with its variable contract "
                f"({canonical_target}: {state_units!r})"
            )
        return canonical_target, state_units, f"{state_units}{self._PER_SECOND_SUFFIX}"

    def _convert_declared_flux(
        self,
        flux: Tensor,
        flux_key: str,
        target_key: str,
        state: Optional[StateDict],
        unit_converter: Any,
        conversion_context_cls: Any,
        conversion_info: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Convert a transport flux from its declared unit to its target's rate unit.

        The source unit is the declaration ``units_map[flux_key]`` read from the
        state's ``units``, ``output_units`` or ``flux_units`` mapping; the target
        unit is ``<contract udunits> s-1`` of the target's variable contract. The
        UnitConverter selects the transform for the pair and checks dimensions.
        A volumetric content rate uses the UnitConverter soil-moisture content
        rule with the layer thickness declared (``layer_thickness_variable``) by
        the contract of the volumetric side and read pointwise from the state.
        Names never select, skip or default a conversion.

        Raises:
            ValueError: the flux declares no unit (including ``state=None``), the
                target has no contract, a declared target unit conflicts with the
                contract, the declared unit is unknown, the unit pair cannot be
                converted, or a content conversion lacks its declared thickness.
        """
        context, units_map = self._get_unit_conversion_inputs(
            state=state,
            unit_converter=unit_converter,
            flux_key=flux_key,
            target_key=target_key,
        )
        declared_source = units_map.get(flux_key) if units_map else None
        if declared_source is None:
            where = "no state was given" if state is None else (
                "the state's 'units', 'output_units' and 'flux_units' mappings do "
                "not declare it"
            )
            raise ValueError(
                f"Transport flux {flux_key!r} -> {target_key!r} declares no units "
                f"({where}); a flux is converted only from its declared unit"
            )
        source_units = str(declared_source).strip()
        canonical_target, state_units, target_units = self._declared_target_rate_units(
            target_key,
            units_map,
        )

        from ...data.standards.dimensionality import dimensionality_from_unit

        if dimensionality_from_unit(source_units) is None:
            raise ValueError(
                f"Unknown unit {source_units!r} declared for transport flux {flux_key!r}"
            )
        target_stock_dimension = dimensionality_from_unit(state_units)
        if target_stock_dimension is None:
            raise ValueError(
                f"Unknown udunits {state_units!r} in the variable contract of "
                f"{canonical_target!r}"
            )

        conversion_info.update({
            'original_units': source_units,
            'target_units': target_units,
            'target_units_source': 'variable_contract',
            'target_variable': canonical_target,
        })
        if source_units == target_units:
            conversion_info['converted'] = False
            return flux, conversion_info

        suffix = self._PER_SECOND_SUFFIX
        source_stock = source_units[: -len(suffix)] if source_units.endswith(suffix) else None
        content_rule = None
        if source_stock is not None:
            content_rule = unit_converter.UNIT_CONVERSIONS.get(
                ("soil_moisture_content", source_stock, state_units)
            )

        if content_rule is not None:
            # References: Hillel, D. (1998), Environmental Soil Physics; CF-1.8
            #   (volume_fraction_of_water_in_soil); UDUNITS-2.
            # Mathematical expression: d(m_w)/dt = (d theta/dt) * L * rho_w
            #   (volumetric -> mass) or d(theta)/dt = (d m_w/dt) / (L * rho_w)
            #   (mass -> volumetric), elementwise over cells.
            # Parameter meanings: theta volumetric water content; m_w areal water
            #   mass; L thickness of the layer the volumetric variable describes;
            #   rho_w water density; the factor comes from the UnitConverter
            #   soil-moisture content rule applied to the stock labels.
            # Units: d theta/dt [m3 m-3 s-1]; d m_w/dt [kg m-2 s-1]; L [m]; rho_w [kg m-3].
            # Domain/constraints: L and rho_w are constant over the step; L is the
            #   state field named by layer_thickness_variable in the contract of
            #   the volumetric (dimensionless) side, the target when its contract
            #   unit is volumetric and the flux otherwise; no fixed state name,
            #   no default, no averaging; an undeclared or absent L raises. Both
            #   labels share the per-second basis.
            # Sign convention: L, rho_w > 0, so the flux direction is preserved.
            volumetric_variable = (
                canonical_target
                if target_stock_dimension.is_dimensionless()
                else flux_key
            )
            layer_thickness_variable = self._declared_layer_thickness_variable(
                volumetric_variable
            )
            if layer_thickness_variable is None:
                raise ValueError(
                    f"Transport flux {flux_key!r} -> {canonical_target!r} "
                    f"({source_units} -> {target_units}) needs the layer thickness "
                    f"of {volumetric_variable!r}, whose variable contract declares "
                    "no layer_thickness_variable"
                )
            # The UnitConverter reads the declared thickness pointwise from the
            # state and raises when the field is absent.
            layer_context = unit_converter.get_conversion_context(
                state,
                layer_thickness_variable=layer_thickness_variable,
            )
            converted_flux, conversion_info = self._convert_flux_using_unit_converter(
                flux=flux,
                context=layer_context,
                unit_converter=unit_converter,
                conversion_context_cls=conversion_context_cls,
                source_units=source_stock,
                target_units=state_units,
                variable_name="soil_moisture_content",
                flux_kind=f"{flux_key} -> {canonical_target}",
                default_conversion_type=content_rule,
                conversion_info=conversion_info,
            )
            conversion_info.update({
                'original_units': source_units,
                'target_units': target_units,
                'layer_thickness_variable': layer_thickness_variable,
                'layer_thickness_used': torch.as_tensor(layer_context.soil_depth).detach(),
                'water_density_used': torch.as_tensor(layer_context.water_density).detach(),
            })
            return converted_flux, conversion_info

        # References: UDUNITS-2 (Unidata); CF Conventions 1.8, Section 3.1.
        # Mathematical expression: F_target = U(F_source; u_flux -> u_target | context)
        #   with u_target = u_state s-1, so that delta = F_target * dt is in u_state.
        # Parameter meanings: F_source flux in its declared unit u_flux; u_state
        #   udunits of the target's variable contract; U the UnitConverter
        #   transform selected for the flux variable and the unit pair (for
        #   example the CF liquid-water-equivalent rule F = rho_w * d for a
        #   depth rate of a water-flux variable, with rho_w the physical-constant
        #   SSOT water.density_liquid; no state density enters that rule).
        # Units: F_source [u_flux]; F_target [u_state s-1]; dt [s].
        # Domain/constraints: a pair the UnitConverter cannot convert raises; the
        #   factor must be linear (checked by _convert_flux_using_unit_converter).
        # Sign convention: positive factor, flux direction preserved.
        return self._convert_flux_using_unit_converter(
            flux=flux,
            context=context,
            unit_converter=unit_converter,
            conversion_context_cls=conversion_context_cls,
            source_units=source_units,
            target_units=target_units,
            variable_name=flux_key,
            flux_kind=(
                f"{flux_key} -> {canonical_target} (contract unit {state_units}; "
                "increment = flux * dt [s])"
            ),
            default_conversion_type=f"{flux_key}: {source_units} -> {target_units}",
            conversion_info=conversion_info,
        )

    def _convert_flux_using_unit_converter(
        self,
        flux: Tensor,
        context: Optional[Any],
        unit_converter: Any,
        conversion_context_cls: Any,
        source_units: str,
        target_units: str,
        variable_name: str,
        flux_kind: str,
        default_conversion_type: str,
        conversion_info: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Convert flux through authoritative unit_converter with metadata update.

        The conversion factor is evaluated once by the UnitConverter with the
        conversion context and applied as a tensor, so a pointwise context field
        the selected rule uses (for example a declared layer thickness) gives a
        per-cell factor and the result stays differentiable with respect to the
        flux and to that field. Rules with constant factors (UDUNITS scales, the
        liquid-water-equivalent rule with the SSOT water density) give a scalar.
        """
        if context is None:
            context = conversion_context_cls()

        if not unit_converter.can_convert(
            source_units, target_units, variable_name=variable_name
        ):
            raise ValueError(
                f"Unsupported {flux_kind} flux units: {source_units} -> {target_units}"
            )

        # References: UDUNITS-2 (Unidata); CF Conventions 1.8, Section 3.1;
        #   BIPM SI Brochure, 9th edition (2019), Section 2.3.
        # Mathematical expression: F_target = f * F_source with
        #   f = U(1; source -> target | context) and U(0; source -> target | context) = 0.
        # Parameter meanings: U is the UnitConverter transform selected for
        #   (variable_name, source_units, target_units); f is its factor evaluated
        #   with the conversion context (pointwise fields the rule reads, such as
        #   a layer thickness, and physical-constant SSOT values).
        # Units: F_source [source_units]; F_target [target_units]; f [target_units / source_units].
        # Domain/constraints: a flux conversion must be linear; a non-zero offset
        #   U(0) raises. f is broadcast elementwise against F_source (no mean, no
        #   scalar cast); incompatible shapes raise in the multiplication.
        # Sign convention: f > 0, so the flux direction is preserved.
        unit_value = torch.ones((), dtype=flux.dtype, device=flux.device)
        factor_tensor = unit_converter.convert(
            unit_value,
            variable_name,
            source_units,
            target_units,
            context,
        )
        offset_tensor = unit_converter.convert(
            torch.zeros_like(unit_value),
            variable_name,
            source_units,
            target_units,
            context,
        )
        if not isinstance(factor_tensor, Tensor) or not isinstance(offset_tensor, Tensor):
            raise TypeError(
                f"UnitConverter must return tensors for {flux_kind} flux "
                f"{source_units} -> {target_units}"
            )
        if bool(torch.any(offset_tensor.detach() != 0)):
            raise ValueError(
                f"{flux_kind} flux conversion {source_units} -> {target_units} is not "
                "linear (non-zero offset); a flux cannot be converted affinely"
            )
        converted_flux = flux * factor_tensor

        conversion_type = unit_converter.UNIT_CONVERSIONS.get(
            (variable_name, source_units, target_units)
        )
        if conversion_type is None:
            conversion_type = default_conversion_type

        factor_record = factor_tensor.detach()
        conversion_info.update({
            'converted': True,
            'conversion_type': conversion_type,
            'original_units': source_units,
            'target_units': target_units,
            # Full (possibly per-cell) factor tensor as applied, plus an explicitly
            # labelled range summary for readable diagnostics.
            'conversion_factor': factor_record,
            'conversion_factor_min': float(factor_record.min().cpu()),
            'conversion_factor_max': float(factor_record.max().cpu()),
        })
        return converted_flux, conversion_info

    def _apply_flux_unit_conversion(
        self,
        flux: Tensor,
        flux_key: str,
        target_key: str,
        state: Optional[StateDict] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Apply unit conversion to flux if needed for consistency with target state variable.

        References:
            - CF-1.8 Climate and Forecast Metadata Conventions
            - UDUNITS-2 Unit Conversion System
            - IPCC AR6 Carbon Flux Conventions

        Mathematical Expressions:
            F_target = U(F_source; u_flux -> u_state s-1 | context), where u_flux
                is the declared unit of the flux key and u_state the udunits of
                the target's variable contract, so delta = F_target * dt [s]
                is in u_state. A volumetric content rate uses the content rule
                d m_w/dt = (d theta/dt) * L * rho_w with L the state field named
                by layer_thickness_variable in the volumetric side's contract.
            Every factor is a tensor applied elementwise (no scalar cast).
            Names never select or skip a conversion; an undeclared flux unit
            (including state=None), a target without a contract, or an
            unconvertible pair raises ValueError.

        Parameters:
            flux : Tensor
                Original flux tensor
            flux_key : str
                Key identifying the flux type
            target_key : str
                Target state variable name
            state : Optional[StateDict]
                Current state for context extraction

        Returns:
            Tuple[Tensor, Dict]: Converted flux and conversion information
        """
        conversion_info = self._initialize_flux_conversion_info(
            flux_key=flux_key,
            target_key=target_key,
        )

        from ...data.standards.unit_converter import unit_converter, ConversionContext

        return self._convert_declared_flux(
            flux=flux,
            flux_key=flux_key,
            target_key=target_key,
            state=state,
            unit_converter=unit_converter,
            conversion_context_cls=ConversionContext,
            conversion_info=conversion_info,
        )

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """
        Primary computation entry point for transport functors.

        Concrete transport functors must implement this method directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward(state, time, dt, **kwargs)."
        )

__all__ = ['TransportMechanismFunctor']
