"""
Soil Water Retention Diagnostic for RELIC

Maps the volumetric water content of the effective soil layer to its soil
water potential through the van Genuchten retention law owned by
relic.utils.soil_hydraulics. The process is an instantaneous physical-field
diagnostic: it advances no water storage and repeats no soil-water flux law.

References
----------
- van Genuchten, M.T. (1980). A closed-form equation for predicting the
  hydraulic conductivity of unsaturated soils. Soil Science Society of America
  Journal, 44, 892-898.
- Hillel, D. (1998). Environmental Soil Physics. Academic Press (pressure
  potential psi = rho_w g h of a pressure head h).

Mathematical Expressions
------------------------
    Se  = (theta - theta_r) / (theta_s - theta_r)
    h   = -(Se^(-1/m) - 1)^(1/n) / alpha,   m = 1 - 1/n
    psi = rho_w g h

Units
-----
theta, theta_r, theta_s [m3 m-3]; Se, n, m [1]; alpha [m-1]; h [m];
rho_w [kg m-3]; g [m s-2]; psi [Pa], returned in MPa.
"""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import (
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.utils.physical_constants import get_physical_constant
from relic.utils.soil_hydraulics import van_genuchten_pressure_head


class SoilWaterRetention(PhysicalFieldFunctor):
    """Soil water potential of the effective soil layer from its water content.

    A point model of the PhysicalField contract: the field is the soil water
    potential of one grid cell, read from the retention curve at the current
    volumetric water content. The retention law itself is the shared utility
    van_genuchten_pressure_head; this class only binds its parameters and the
    head-to-potential conversion.

    An explicit provider selection declares a required or lagged relation.
    With no selected provider the existing external-state or graph-linked
    behavior is retained. The declaration changes neither the retention law
    nor the value, time, or phase of its committed-state read.
    """

    SPATIAL_DIMS = None
    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    NAMESPACE = "hydrosphere.soil_moisture"

    def __init__(
        self,
        name: str = "soil_water_retention",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        source_config = config if config is not None else {}
        provider = source_config.get("soil_moisture_provider")
        lagged = source_config.get("soil_moisture_lagged", False)
        if provider is not None:
            if not isinstance(provider, str):
                raise TypeError(
                    "soil_moisture_provider must be a non-empty Functor name or None"
                )
            if not provider or provider != provider.strip() or provider == name:
                raise ValueError(
                    "soil_moisture_provider must be a trimmed non-empty name "
                    "different from the retention Functor"
                )
        if type(lagged) is not bool:
            raise TypeError("soil_moisture_lagged must be a bool")
        if lagged and provider is None:
            raise ValueError(
                "soil_moisture_lagged requires an explicit soil_moisture_provider"
            )

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Soil water potential of the effective soil layer from its volumetric "
                "water content through the van Genuchten retention law"
            ),
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "Se = (theta - theta_r) / (theta_s - theta_r); "
                "h = -(Se^(-1/m) - 1)^(1/n) / alpha, m = 1 - 1/n; psi = rho_w g h"
            ),
            references=[
                "van Genuchten, M.T. (1980). Soil Science Society of America Journal, 44, 892-898",
                "Hillel, D. (1998). Environmental Soil Physics. Academic Press",
            ],
            parameters={
                "hydrosphere.soil_moisture.alpha_vg": (
                    "Van Genuchten alpha parameter [m-1]"
                ),
                "hydrosphere.soil_moisture.n_vg": (
                    "Van Genuchten n parameter [dimensionless]"
                ),
                "hydrosphere.soil_moisture.residual_water_content": (
                    "Residual water content [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.porosity": (
                    "Total soil porosity (saturated water content) [m3 m-3]"
                ),
            },
            input_variables=["soil_moisture"],
            output_variables=["soil_water_potential"],
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(name=name, config=config or {}, metadata=metadata, device=device)
        self._soil_moisture_provider = provider
        self._soil_moisture_lagged = lagged

    def get_dependencies(self) -> DependencyDeclaration:
        """Declare only the explicitly selected soil-water provider relation."""
        # References: architecture_overview.md Sections4.3 and4.4.1.
        # Mathematical expression: D = merge(D_selected, D_instance), where
        # D_selected is empty, required(provider), or lagged(provider).
        # Parameter meanings: provider names the declared theta writer; lagged
        # acknowledges its committed-state read in the existing group schedule.
        # Units: dependency names and flags are dimensionless; theta remains
        # soil_moisture [m3 m-3] and potential remains [MPa].
        # Domain/constraints: the constructor validates the explicit selection;
        # no source is inferred from state values and no theta is cached.
        # Sign convention: this declaration adds no flux, source, sink, state
        # advance, time shift, or change of physical sampling phase.
        provider = self._soil_moisture_provider
        if provider is None:
            return self._merge_declared_dependencies()
        category = "lagged" if self._soil_moisture_lagged else "required"
        return self._merge_declared_dependencies({category: [provider]})

    def get_provides(self) -> list:
        """Define output variables provided by this Functor."""
        return ["soil_water_potential"]

    def get_value_at(self, coordinates: Tensor) -> Tensor:
        """A point model holds no spatial field to sample."""
        raise NotImplementedError(
            "SoilWaterRetention is a point model; the potential of the cell is "
            "the declared output soil_water_potential"
        )

    def forward(self, state: StateDict, time: float, dt: float, **kwargs) -> FluxDict:
        """Return the soil water potential of the current soil water content."""
        # References: van Genuchten (1980), retention curve with m = 1 - 1/n;
        #   Hillel (1998), pressure potential of a pressure head.
        # Mathematical expression:
        #   Se = (theta - theta_r) / (theta_s - theta_r);
        #   h = -(Se^(-1/m) - 1)^(1/n) / alpha, m = 1 - 1/n;
        #   psi = rho_w g h.
        # Parameter meanings: theta volumetric water content; theta_r residual and
        #   theta_s saturated water content; Se effective saturation; alpha and n
        #   van Genuchten shape parameters; h pressure head; rho_w liquid water
        #   density; g gravitational acceleration; psi soil water potential.
        # Units: theta, theta_r, theta_s [m3 m-3]; Se, n, m [1]; alpha [m-1];
        #   h [m]; rho_w [kg m-3]; g [m s-2]; psi [Pa] converted to [MPa].
        # Domain/constraints: finite floating theta with theta_r <= theta <= theta_s
        #   (outside it the retention law is undefined and the call raises);
        #   theta_s > theta_r, alpha > 0 and n > 1. The inversion and its
        #   saturation floor are those of van_genuchten_pressure_head. No state is
        #   advanced: this read-only map has no source, sink or residual.
        # Sign convention: h and psi are relative to free water, zero at
        #   saturation and negative under tension.
        _ = (time, dt, kwargs)
        if "soil_moisture" not in state:
            raise KeyError("soil_moisture is required")
        theta = state["soil_moisture"]
        if not isinstance(theta, Tensor):
            raise TypeError("soil_moisture must be a torch.Tensor")
        if not torch.is_floating_point(theta):
            raise TypeError("soil_moisture must be a floating torch.Tensor")
        if not bool(torch.isfinite(theta).all().item()):
            raise ValueError("soil_moisture must contain only finite values")

        target = {"device": theta.device, "dtype": theta.dtype}
        alpha = torch.as_tensor(
            self.get_parameter_value("alpha_vg", namespace=self.NAMESPACE, required=True),
            **target,
        )
        n_value = torch.as_tensor(
            self.get_parameter_value("n_vg", namespace=self.NAMESPACE, required=True),
            **target,
        )
        theta_r = torch.as_tensor(
            self.get_parameter_value(
                "residual_water_content", namespace=self.NAMESPACE, required=True
            ),
            **target,
        )
        theta_s = torch.as_tensor(
            self.get_parameter_value("porosity", namespace=self.NAMESPACE, required=True),
            **target,
        )
        if not bool(((theta_s > theta_r) & (alpha > 0) & (n_value > 1)).all().item()):
            raise ValueError(
                "retention parameters must satisfy porosity > residual_water_content, "
                "alpha_vg > 0 and n_vg > 1"
            )
        if bool(((theta < theta_r) | (theta > theta_s)).any().item()):
            raise ValueError(
                "soil_moisture must lie within [residual_water_content, porosity]"
            )

        pressure_head = van_genuchten_pressure_head(
            theta, alpha, n_value, theta_r, theta_s
        )
        water_density = torch.as_tensor(get_physical_constant("water_density"), **target)
        gravity = torch.as_tensor(get_physical_constant("gravity_acceleration"), **target)
        potential = unit_converter.convert(
            pressure_head * gravity * water_density,
            variable_name="soil_water_potential",
            source_units="Pa",
            target_units="MPa",
        )

        return {
            "soil_water_potential": UpdateItem(
                value=potential,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="MPa",
                source_functor=self.name,
            )
        }


__all__ = ["SoilWaterRetention"]
