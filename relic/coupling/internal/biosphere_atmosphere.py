"""
Leaf boundary-layer exchange between canopy air and the biosphere.

This module owns one cross-scale operation: conversion of the canonical surface
friction velocity into the water-vapor boundary-layer conductance of a
representative leaf. Stomatal
conductance, photosynthesis, evapotranspiration, and energy balance remain with
their dedicated scientific Functors.

References:
    - CTSM Technical Note, Eqs. 2.5.128 and 2.5.135, bulk leaf wind and
      leaf boundary-layer resistance.
    - Campbell, G.S. and Norman, J.M. (1998), Environmental Biophysics.
    - Bonan, G.B. et al. (2014), Geosci. Model Dev. 7, 2193-2222.

Mathematical expressions:
    U_av = u_*
    r_b = (1 / C_v) * (u_* / d_leaf)^(-1/2)
    g_b,ms = C_v * sqrt(U_av / d_leaf)
    g_b,mol = g_b,ms * P / (R * T)

Parameter meanings and units:
    u_*: canonical friction velocity from atmospheric turbulence [m s-1].
    d_leaf: characteristic leaf dimension in the wind direction [m].
    C_v: turbulent transfer coefficient [m s-0.5].
    P: atmospheric pressure [Pa].
    R: universal gas constant from the physical-constants SSOT [J mol-1 K-1].
    T: air temperature [K].

Open-system route:
    The result is an instantaneous transport coefficient, not a material flux
    or state increment. Gas-exchange Functors consume it to construct explicit
    atmosphere-biosphere matter ports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.state.types import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import (
    ConversionContext,
    unit_converter,
)


PARAMETER_NAMESPACE = "coupling.scales.biosphere_atmosphere"
DEFAULT_TURBULENCE_DEPENDENCY = "atmospheric_turbulence"
INPUT_VARIABLES = (
    "friction_velocity",
    "air_temperature",
    "atmospheric_pressure",
)
OUTPUT_VARIABLE = "boundary_layer_conductance"

MATHEMATICAL_EXPRESSION = (
    "U_av = u_*; "
    "r_b = (1 / C_v) * (u_* / d_leaf)^(-1/2); "
    "g_b,ms = C_v * sqrt(u_* / d_leaf); "
    "g_b,mol = g_b,ms * P / (R * T)"
)

REFERENCES = [
    "CTSM Technical Note, Eqs. 2.5.128 and 2.5.135, bulk leaf wind and leaf boundary-layer resistance",
    "Campbell, G.S. and Norman, J.M. (1998). Environmental Biophysics",
    "Bonan, G.B. et al. (2014). Geosci. Model Dev. 7, 2193-2222",
]


class BiosphereAtmosphereCoupling(ScaleCouplingFunctor):
    """Map friction velocity to bulk leaf boundary-layer conductance."""

    def __init__(
        self,
        name: str = "biosphere_atmosphere",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[FunctorMetadata] = None,
        parameter_manager: Optional[Any] = None,
    ) -> None:
        raw_config = {} if config is None else config
        if not isinstance(raw_config, dict):
            raise ValueError("Config must be a dictionary")
        if "parameters" in raw_config:
            raise ValueError(
                "Biosphere-atmosphere parameters must be supplied through "
                "ParameterManager YAML."
            )
        allowed_config_keys = {"upstream_turbulence_functor", "dependencies"}
        unknown_keys = set(raw_config) - allowed_config_keys
        if unknown_keys:
            raise ValueError(
                "Unsupported biosphere-atmosphere config keys: "
                + ", ".join(sorted(unknown_keys))
            )

        dependency = raw_config.get(
            "upstream_turbulence_functor", DEFAULT_TURBULENCE_DEPENDENCY
        )
        if not isinstance(dependency, str) or not dependency.strip():
            raise ValueError("upstream_turbulence_functor must be a non-empty string")

        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                version="2.0.0",
                description=(
                    "CTSM single-layer bulk leaf boundary conductance from "
                    "canonical friction velocity"
                ),
                category=ProcessCategory.SCALE_COUPLING,
                execution_phase=ExecutionPhase.COUPLING,
                mathematical_expression=MATHEMATICAL_EXPRESSION,
                references=list(REFERENCES),
                parameters={
                    "leaf_characteristic_dimension": (
                        "Leaf dimension in the wind direction [m]"
                    ),
                    "boundary_layer_transfer_coefficient": (
                        "CTSM turbulent transfer coefficient [m s-0.5]"
                    ),
                },
                input_variables=list(INPUT_VARIABLES),
                output_variables=[OUTPUT_VARIABLE],
                conservation_laws=[],
                spatial_dimensions=0,
                temporal_scale=TemporalScale.SECOND,
                coupling_interfaces=[
                    "atmospheric_turbulence",
                    "leaf_gas_exchange",
                ],
                spatial_scale=SpatialScale.PATCH,
                variable_kind=VariableKind.DIAGNOSTIC,
            )

        # config["dependencies"] seeds the instance dependencies through
        # FunctorBase; the upstream turbulence provider is added to them.
        super().__init__(
            name=name,
            config={
                key: raw_config[key] for key in ("dependencies",) if key in raw_config
            },
            metadata=metadata,
            parameter_manager=parameter_manager,
            source_scale="canopy_air",
            target_scale="leaf_surface",
            coupling_type="spatial",
        )
        self.add_dependency(dependency.strip())
        self.unit_converter = unit_converter

        leaf_dimension = self._load_scalar_parameter(
            "leaf_characteristic_dimension", units="m"
        )
        transfer_coefficient = self._load_scalar_parameter(
            "boundary_layer_transfer_coefficient", units="m s-0.5"
        )
        self.register_buffer(
            "leaf_characteristic_dimension",
            torch.tensor(leaf_dimension, dtype=torch.float64),
        )
        self.register_buffer(
            "boundary_layer_transfer_coefficient",
            torch.tensor(transfer_coefficient, dtype=torch.float64),
        )

    def _load_scalar_parameter(self, key: str, *, units: str) -> float:
        value = self.get_parameter_value(
            key,
            namespace=PARAMETER_NAMESPACE,
            units=units,
            source_order=("manager",),
            required=True,
        )
        tensor = torch.as_tensor(value)
        if tensor.numel() != 1 or not tensor.is_floating_point():
            raise TypeError(f"{key} must be a floating-point scalar")
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"{key} must be finite")
        scalar = float(tensor.detach().reshape(()).cpu())
        if scalar <= 0.0:
            raise ValueError(f"{key} must be positive")
        return scalar

    @staticmethod
    def _require_inputs(
        state: Mapping[str, Any],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        missing = [key for key in INPUT_VARIABLES if key not in state]
        if missing:
            raise ValueError(
                "Missing required biosphere-atmosphere inputs: "
                + ", ".join(missing)
            )

        tensors: List[Tensor] = []
        reference: Optional[Tensor] = None
        for key in INPUT_VARIABLES:
            value = state[key]
            if not isinstance(value, Tensor):
                raise TypeError(f"{key} must be a torch.Tensor")
            if not value.is_floating_point():
                raise TypeError(f"{key} must use a floating-point dtype")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{key} must contain only finite values")
            if bool(torch.any(value <= 0.0)):
                raise ValueError(f"{key} must be strictly positive")
            if reference is None:
                reference = value
            elif value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(
                    "Biosphere-atmosphere inputs must share dtype and device"
                )
            tensors.append(value)

        try:
            broadcast = torch.broadcast_tensors(*tensors)
        except RuntimeError as exc:
            raise ValueError(
                "Biosphere-atmosphere inputs must have broadcast-compatible shapes"
            ) from exc
        return broadcast[0], broadcast[1], broadcast[2]

    def forward(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        """Compute the instantaneous leaf boundary-layer conductance.

        References:
            CTSM Technical Note Eqs. 2.5.128 and 2.5.135; Campbell and
            Norman (1998); Bonan et al. (2014).
        Mathematical expression:
            U_av = u_* for the CTSM single-layer bulk canopy closure.
            g_b,ms = C_v sqrt(u_* / d_leaf)
            g_b,mol = g_b,ms P / (R T)
        Parameter meanings:
            u_* is canonical friction velocity [m s-1], d_leaf is leaf
            dimension [m], C_v is transfer coefficient [m s-0.5], P is
            pressure [Pa], and T is air temperature [K].
        Units and route:
            The output is mol m-2 leaf s-1. It is an instantaneous diagnostic
            transport coefficient; no mass or energy state is advanced here.
        """
        # References: CTSM Technical Note Eqs. 2.5.128 and 2.5.135; Campbell and
        #   Norman (1998); Bonan et al. (2014); unit_converter conductance_ms_to_mol.
        # Mathematical expression: U_av = u_*; g_b,ms = C_v * sqrt(u_* / d_leaf);
        #   g_b,mol = g_b,ms * P / (R * T).
        # Parameter meanings: u_* canonical friction velocity; d_leaf leaf dimension in
        #   the wind direction; C_v turbulent transfer coefficient; P air pressure;
        #   T air temperature; R universal gas constant (SSOT); g_b conductance.
        # Units: u_* [m s-1]; d_leaf [m]; C_v [m s-0.5]; P [Pa]; T [K];
        #   R [J mol-1 K-1]; g_b,ms [m s-1]; output boundary_layer_conductance
        #   [mol m-2 s-1] per unit leaf area, REPLACE UpdateItem, per_timestep=False.
        # Domain/constraints: the three inputs must be present, floating-point, finite,
        #   strictly positive, same dtype/device and broadcast-compatible; d_leaf and
        #   C_v are positive finite YAML scalars; output must be finite; no state advance.
        # Sign convention: not applicable: g_b is a non-negative instantaneous transport
        #   coefficient, not a directed flux; no source or sink term is created here.
        del time, dt, kwargs
        friction_velocity, temperature, pressure = self._require_inputs(state)
        leaf_dimension = self.leaf_characteristic_dimension.to(
            device=friction_velocity.device, dtype=friction_velocity.dtype
        )
        transfer_coefficient = self.boundary_layer_transfer_coefficient.to(
            device=friction_velocity.device, dtype=friction_velocity.dtype
        )

        # References: CTSM Technical Note Eqs. 2.5.128 and 2.5.135.
        # Expression: U_av = u_*; g_b,ms = C_v * sqrt(u_* / d_leaf).
        # Parameters: u_* [m s-1], d_leaf [m], C_v [m s-0.5].
        # Route: diagnostic transport coefficient; no local source or sink.
        incident_leaf_wind_speed = friction_velocity
        conductance_ms = transfer_coefficient * torch.sqrt(
            incident_leaf_wind_speed / leaf_dimension
        )
        conductance_mol = self.unit_converter.convert(
            conductance_ms,
            OUTPUT_VARIABLE,
            "m s-1",
            "mol m-2 s-1",
            ConversionContext(
                temperature=temperature,
                pressure=pressure,
            ),
        )
        conductance_mol = torch.as_tensor(
            conductance_mol,
            device=friction_velocity.device,
            dtype=friction_velocity.dtype,
        )
        if not bool(torch.all(torch.isfinite(conductance_mol))):
            raise ValueError("boundary_layer_conductance must be finite")

        return {
            OUTPUT_VARIABLE: UpdateItem(
                value=conductance_mol,
                mode=UpdateMode.REPLACE,
                unit="mol m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            )
        }


__all__ = ["BiosphereAtmosphereCoupling"]
