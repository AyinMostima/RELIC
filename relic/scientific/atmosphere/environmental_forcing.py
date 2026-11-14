"""Canonical atmospheric vapor-pressure-deficit forcing for RELIC.

The Functor consumes externally supplied air temperature and relative humidity
and derives one canonical vapor pressure deficit in Pa. It does not own, echo,
or synthesize the external forcing fields.

Mathematical Framework
----------------------
1. Temperature conversion: T_C = T_K - kelvin_offset
2. Saturation vapor pressure (Magnus form):
   es = es_coeff * exp(a * (T - kelvin_offset) / (T - kelvin_offset + b))
3. Vapor pressure deficit in Pa:
   VPD_Pa = kPa_to_Pa(max(es * (1 - RH), 0))

Units
-----
- Air temperature: K
- Relative humidity: dimensionless
- Vapor pressure deficit: Pa
"""

from typing import Any

import torch
from torch import Tensor

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import ConversionContext, unit_converter
from relic.utils.physical_constants import get_physical_constant


class EnvironmentalForcingFunctor(PhysicalFieldFunctor):
    """Derive atmospheric VPD from required external forcing inputs."""
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "environmental_forcing",
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description="Derive canonical atmospheric VPD in Pa from external forcing",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.PREPROCESSING,
            mathematical_expression=(
                "VPD_Pa = kPa_to_Pa(max(es0 * exp(a*T_C/(T_C+b)) "
                "* (1-RH), 0)); T_C = T_K - T0"
            ),
            references=[
                "Allen et al. (1998) FAO Irrigation and Drainage Paper 56",
                "Monteith & Unsworth (2008) Principles of Environmental Physics",
                "Campbell & Norman (1998) Environmental Biophysics",
            ],
            parameters={
                "atmosphere.environmental_forcing.vpd.es_coefficient": (
                    "Magnus coefficient es0 [kPa]"
                ),
                "atmosphere.environmental_forcing.vpd.a_coefficient": (
                    "Magnus coefficient a [dimensionless]"
                ),
                "atmosphere.environmental_forcing.vpd.b_coefficient": (
                    "Magnus denominator coefficient b [degC]"
                ),
            },
            input_variables=["air_temperature", "relative_humidity"],
            output_variables=["vapor_pressure_deficit"],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FORCING,
        )

        # The config goes to FunctorBase, the single reader of
        # config["dependencies"] and config parameters.
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            device=device,
        )
        self.optional_input_variables: set[str] = set()
        self._initialize_parameters()

    def _fetch_required_parameter(
        self,
        key: str,
        namespace: str,
    ) -> float:
        """Fetch a required scalar parameter with validation."""
        value = self.get_parameter_value(
            key,
            namespace=namespace,
            required=True,
        )
        if isinstance(value, Tensor):
            value = value.detach().reshape(-1)[0].tolist()
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Parameter {namespace}.{key} must be numeric.") from exc

    def _initialize_parameters(self) -> None:
        """Load empirical coefficients and the physical-constant SSOT value."""
        forcing_ns = "atmosphere.environmental_forcing"
        self.celsius_to_kelvin = float(get_physical_constant("celsius_to_kelvin"))
        self.es_coefficient = self._fetch_required_parameter(
            "vpd.es_coefficient",
            forcing_ns,
        )
        self.vpd_a = self._fetch_required_parameter(
            "vpd.a_coefficient",
            forcing_ns,
        )
        self.vpd_b = self._fetch_required_parameter(
            "vpd.b_coefficient",
            forcing_ns,
        )
        self._unit_converter = unit_converter
        self._conversion_context = ConversionContext()

    def _require_input(self, state: dict[str, Tensor], key: str) -> Tensor:
        """Return one finite, non-empty floating tensor from external state."""
        if key not in state:
            raise KeyError(f"Missing required state input '{key}'.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"State '{key}' must be a torch.Tensor.")
        if value.numel() == 0 or not torch.isfinite(value).all():
            raise ValueError(f"State '{key}' must be finite and non-empty.")
        if not value.is_floating_point():
            raise TypeError(f"State '{key}' must use a floating-point dtype.")
        return value.to(device=self.device)

    def forward(
        self,
        state: dict[str, Tensor],
        time: float,
        dt: float,
        **kwargs,
    ) -> dict[str, UpdateItem]:
        """
        Derive canonical atmospheric vapor pressure deficit.

        Parameters
        ----------
        state : Dict[str, Tensor]
            Required ``air_temperature`` [K] and ``relative_humidity`` [1].
        time : float
            Current simulation time [s].
        dt : float
            Time step [s].

        Returns
        -------
        Dict[str, UpdateItem]
            One replacement update, ``vapor_pressure_deficit`` [Pa].
        """
        air_temperature = self._require_input(state, "air_temperature")
        relative_humidity = self._require_input(state, "relative_humidity")
        if torch.any((relative_humidity < 0) | (relative_humidity > 1)):
            raise ValueError("State 'relative_humidity' must be within [0, 1].")

        # References:
        #   Allen et al. (1998), FAO Irrigation and Drainage Paper 56, Eq. 11.
        #   Monteith and Unsworth (2008), Principles of Environmental Physics.
        # Mathematical expression:
        #   T_C = T_K - T0
        #   e_s,kPa = e_s0 * exp(a * T_C / (T_C + b))
        #   VPD_Pa = kPa_to_Pa(max(e_s,kPa * (1 - RH), 0))
        # Parameter meanings:
        #   T_K: external air temperature [K]
        #   T0: Celsius-to-Kelvin offset from physical constants SSOT [K]
        #   e_s0: saturation vapor pressure coefficient [kPa]
        #   a: Magnus slope coefficient [dimensionless]
        #   b: Magnus temperature coefficient [degC]
        #   RH: external relative humidity [dimensionless]
        # Open-system route:
        #   T_K and RH enter through external forcing ports; VPD is a derived
        #   observation and does not advance material or energy storage.
        temperature_c = air_temperature - self.celsius_to_kelvin
        saturation_vapor_pressure_kpa = self.es_coefficient * torch.exp(
            self.vpd_a * temperature_c / (temperature_c + self.vpd_b)
        )
        vapor_pressure_deficit_kpa = torch.clamp(
            saturation_vapor_pressure_kpa * (1.0 - relative_humidity),
            min=0.0,
        )
        vapor_pressure_deficit_pa = self._unit_converter.kpa_to_pa(
            vapor_pressure_deficit_kpa,
            self._conversion_context,
        )
        if not torch.isfinite(vapor_pressure_deficit_pa).all():
            raise ValueError(
                "Computed 'vapor_pressure_deficit' must be finite for the "
                "supplied air_temperature and relative_humidity."
            )

        return {
            "vapor_pressure_deficit": UpdateItem(
                value=vapor_pressure_deficit_pa,
                mode=UpdateMode.REPLACE,
                unit="Pa",
                per_timestep=True,
                source_functor=self.name,
            )
        }
