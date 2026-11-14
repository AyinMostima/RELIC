"""Well-mixed boundary-layer source of the surface isoprene flux.

References:
    Stull, R. B. (1988), An Introduction to Boundary Layer Meteorology.
    Vila-Guerau de Arellano, J. et al. (2015), Atmospheric Boundary Layer:
    Integrating Air Chemistry and Land Interactions.
    U.S. EPA (2019), CMAQ User's Guide, emissions conversion.

Mathematical expression:
    d(chi)/dt = F / (h n), n = P / (R T)
    Delta chi = F dt R T / (P h)

Parameter meanings and units:
    F is the surface isoprene molar flux [mol m-2 s-1, ground area].
    h is the planetary boundary-layer height [m].
    n is the air molar density [mol m-3].
    P is atmospheric pressure [Pa] and T is air temperature [K].
    R is the universal gas constant [J mol-1 K-1].
    chi is the isoprene mixing ratio [mol mol-1].

Open-system route:
    Open mass transfer across the lower boundary of the well-mixed box. The
    column burden gained by the box equals the molar amount that crosses the
    boundary; the law itself is the existing SpeciesMapper.flux_to_mixing_ratio.
"""
from functools import reduce
import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from ....core.abstractions import FluxDict, StateDict
from ....core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    TransportMechanismFunctor,
    VariableKind,
)
from ....core.state import UpdateItem, UpdateMode
from ....utils.physical_constants import get_physical_constant
from ...anthroposphere.emissions.species_mapping import SpeciesMapper

logger = logging.getLogger(__name__)

BALANCE_RESIDUAL = 'diagnostics.boundary_layer_emission.relative_column_balance_residual'


class BoundaryLayerEmissionSource(TransportMechanismFunctor):
    """Hand the surface isoprene flux over to the well-mixed boundary layer."""

    TRANSPORT_TYPE = "gas_exchange"
    PHYSICAL_CONSTANT_KEYS = (
        "relic.physical.constants.universal.gas_constant",
    )
    _REQUIRED_INPUTS = (
        'isoprene_emission',
        'pbl_height',
        'air_temperature',
        'atmospheric_pressure',
    )

    def __init__(
        self,
        name: str = 'boundary_layer_emission_source',
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the source; the process has no parameter of its own."""

        metadata = FunctorMetadata(
            name=name,
            version='1.0.0',
            description='Well-mixed boundary-layer source of the surface isoprene flux',
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                '$\\frac{d\\chi}{dt}=\\frac{F}{h\\,n}$, $n=\\frac{P}{R\\,T}$; '
                '$\\Delta\\chi=\\frac{F\\,\\Delta t\\,R\\,T}{P\\,h}$'
            ),
            references=[
                'Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer Academic',
                'Vila-Guerau de Arellano, J. et al. (2015). Atmospheric Boundary Layer: '
                'Integrating Air Chemistry and Land Interactions. Cambridge University Press',
                'Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics, 3rd ed.',
                "U.S. EPA (2019). CMAQ User's Guide, emissions conversion",
            ],
            parameters={},
            input_variables=list(self._REQUIRED_INPUTS),
            output_variables=['isoprene_mixing_ratio', BALANCE_RESIDUAL],
            conservation_laws=['mass'],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=['bvoc_emission', 'chemistry', 'eddy_diffusion_blm'],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        super().__init__(name=name, config=config, metadata=metadata)
        logger.info("Initialized %s '%s'", self.__class__.__name__, name)

    def get_dependencies(self) -> Any:
        """Order after an emission provider when one is registered."""

        # References: DeclaredDependencies contract (relic/core/dependency_graph.py);
        #   precedent ChemistryFunctor (optional atmospheric_radiation).
        # Mathematical expression: Delta chi = F dt R T / (P h) reads F of the same step.
        # Parameter meanings: F is isoprene_emission, written by one of the two providers.
        # Units: F [mol m-2 s-1]; dependency names carry no unit.
        # Domain/constraints: both providers are optional same-step providers; without a
        #   registered provider isoprene_emission must be forcing, and a missing value raises.
        # Sign convention: not applicable (ordering declaration).
        return self._merge_declared_dependencies(
            {'optional': ['megan_bvoc_model', 'mechanistic_isoprene_synthesis']}
        )

    def physical_constant_keys(self) -> tuple[str, ...]:
        """Return physical constants kept outside the domain parameter map."""

        return self.PHYSICAL_CONSTANT_KEYS

    def _aligned_inputs(self, state: StateDict) -> tuple[Tensor, ...]:
        """Return the four finite inputs on one device at their widest dtype."""

        missing = [key for key in self._REQUIRED_INPUTS if key not in state]
        if missing:
            raise KeyError(f'Boundary-layer emission source missing inputs: {missing}')
        tensors = []
        for key in self._REQUIRED_INPUTS:
            value = state[key]
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f'Boundary-layer emission input {key} must be a floating tensor.')
            if bool(torch.any(~torch.isfinite(value))):
                raise ValueError(f'Boundary-layer emission input {key} must be finite.')
            tensors.append(value)
        dtype = reduce(torch.promote_types, (tensor.dtype for tensor in tensors))
        device = tensors[0].device
        return tuple(tensor.to(device=device, dtype=dtype) for tensor in tensors)

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """Return the mixing-ratio increment of the surface flux over one step."""

        # References: Stull (1988); Vila-Guerau de Arellano et al. (2015); CMAQ User's
        #   Guide (2019) through SpeciesMapper.flux_to_mixing_ratio.
        # Mathematical expression:
        #   Delta chi = F dt R T / (P h), equal to F dt / (h n) with n = P / (R T);
        #   r = (Delta chi h n - F dt) / (F dt), and r = 0 where F = 0.
        # Parameter meanings: F surface molar flux, h boundary-layer height, n air molar
        #   density, P pressure, T air temperature, R gas constant, r relative column
        #   molar balance residual.
        # Units: F [mol m-2 s-1], h [m], n [mol m-3], P [Pa], T [K], R [J mol-1 K-1],
        #   dt [s], Delta chi [mol mol-1], r [1].
        # Domain/constraints: all inputs finite; F >= 0, and h, P, T, dt > 0, enforced by
        #   the shared law. F, h, P and T are held at their values in the step.
        # Sign convention: F > 0 is upward, surface to atmosphere, and gives Delta chi > 0.
        emission, height, temperature, pressure = self._aligned_inputs(state)
        increment = SpeciesMapper.flux_to_mixing_ratio(
            molar_flux=emission,
            pressure=pressure,
            temperature=temperature,
            layer_height=height,
            dt=dt,
        )
        gas_constant = emission.new_tensor(get_physical_constant('gas_constant'))
        molar_density = pressure / (gas_constant * temperature)
        boundary_input = emission * dt
        imbalance = increment * height * molar_density - boundary_input
        relative_residual = torch.where(
            boundary_input > 0.0,
            imbalance / torch.where(
                boundary_input > 0.0,
                boundary_input,
                torch.ones_like(boundary_input),
            ),
            torch.zeros_like(imbalance),
        )
        return {
            'isoprene_mixing_ratio': UpdateItem(
                value=increment,
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            BALANCE_RESIDUAL: relative_residual,
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Evaluate the relative column molar ledger of the submitted increment."""

        # References: Stull (1988), mixed-layer budget.
        # Mathematical expression: r = (Delta chi h n - F dt) / (F dt), reported as max|r|.
        # Parameter meanings: Delta chi the submitted increment, h n the column air burden
        #   per mixing ratio, F dt the molar amount crossing the lower boundary.
        # Units: r [1]; the numerator and the denominator are mol m-2.
        # Domain/constraints: a finite non-empty residual tensor; no cancellation across cells.
        # Sign convention: r > 0 means the box gained more than crossed the boundary.
        del input_state
        if BALANCE_RESIDUAL not in output_updates:
            raise KeyError(f"Missing conservation output '{BALANCE_RESIDUAL}'")
        residual = output_updates[BALANCE_RESIDUAL]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError('Boundary-layer emission residual must be a torch.Tensor.')
        if residual.numel() == 0:
            raise ValueError('Boundary-layer emission residual must be non-empty.')
        if not bool(torch.all(torch.isfinite(residual))):
            raise ValueError('Boundary-layer emission residual must be finite.')
        return {'mass': float(torch.max(torch.abs(residual)).detach().item())}

    def get_transport_metadata(self) -> Dict[str, Any]:
        """Return metadata of the single declared source law."""

        return {
            'transport_type': self.transport_type,
            'closure': 'well_mixed_boundary_layer_source',
            'required_state_inputs': self._REQUIRED_INPUTS,
            'open_system_route': 'port',
        }


__all__ = ['BoundaryLayerEmissionSource']
