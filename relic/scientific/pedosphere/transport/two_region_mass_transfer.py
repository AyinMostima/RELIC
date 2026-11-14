"""Two-region (mobile-immobile) solute exchange with exponential memory.

Theory document: docs/source/theory/pedosphere_two_region_mass_transfer.md

The mobile soil-water region exchanges solute with a stagnant immobile
region through first-order mass transfer (van Genuchten and Wierenga 1976;
Haggerty and Gorelick 1995).  Eliminating the immobile concentration turns
the mobile balance into a Volterra equation with the completely monotone
kernel K(t) = alpha omega exp(-omega t), omega = alpha / theta_im, so the
immobile concentration is the hidden realization state of that memory
(revised RELIC foundations, exponential-kernel passive realization).  The
Functor advances both concentrations with the exact solution of the linear
exchange and reports the extended-storage dissipation and the closed-ledger
residual of every step.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from ....core.abstractions import FluxDict, StateDict
from ....core.functor.functor_bridge_utils import NamespaceDefaultMixin
from ....core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    SubFlowAccuracy,
    TemporalScale,
    TransportMechanismFunctor,
    VariableKind,
)
from ....core.state import UpdateItem, UpdateMode


class TwoRegionMassTransferFunctor(NamespaceDefaultMixin, TransportMechanismFunctor):
    """Mobile-immobile first-order exchange (TransportMechanism contract, memory semantics).

    References:
      - van Genuchten, M.Th. & Wierenga, P.J. (1976). Mass transfer studies
        in sorbing porous media I. Soil Sci. Soc. Am. J. 40, 473-480.
      - Haggerty, R. & Gorelick, S.M. (1995). Multiple-rate mass transfer for
        modeling diffusion and surface reactions in media with pore-scale
        heterogeneity. Water Resour. Res. 31, 2383-2400.
      - Revised RELIC foundations, memory/Volterra semantics: exponential
        kernels admit a passive internal realization with extended storage.
    Mathematical expression:
      theta_m dc_m/dt = -alpha (c_m - c_im),  theta_im dc_im/dt = alpha (c_m - c_im);
      m = theta_m c_m + theta_im c_im is conserved and d = c_m - c_im obeys
      d(t + dt) = d(t) exp(-kappa dt), kappa = alpha (theta_m + theta_im)/(theta_m theta_im).
      Memory form: theta_m dc_m/dt = -alpha c_m + alpha e^{-omega t} c_im(0)
      + int_0^t alpha omega e^{-omega (t-s)} c_m(s) ds, omega = alpha/theta_im.
      Extended storage H = 0.5 (theta_m c_m^2 + theta_im c_im^2) with
      dH/dt = -alpha (c_m - c_im)^2 <= 0.
    Parameter meanings:
      c_m, c_im mobile and immobile concentrations; alpha first-order mass
      transfer coefficient; theta_m, theta_im mobile and immobile water
      contents; dt the simulator step.
    Units:
      c kg m-3; alpha s-1; theta m3 m-3; dt s; mass_transfer_rate kg m-3 s-1;
      storage_dissipation kg2 m-6 s-1; memory_residual kg m-3.
    Domain/constraints:
      alpha > 0, theta_m > 0, theta_im > 0 (ParameterManager bounds);
      nonnegative finite concentrations of one shape; finite positive dt;
      the exact update is unconditionally stable and mass conserving.
    Sign convention:
      mass_transfer_rate alpha (c_m - c_im) is positive from the mobile to the
      immobile region; the mobile increment has the opposite sign.
    """

    TRANSPORT_TYPE = "interphase_mass_transfer"

    parameter_namespace = "pedosphere.transport"
    MOBILE_VARIABLE = "mobile_solute_concentration"
    IMMOBILE_VARIABLE = "immobile_solute_concentration"
    STATE_VARIABLES = (MOBILE_VARIABLE, IMMOBILE_VARIABLE)
    EFFORT_STATE_VARIABLE = MOBILE_VARIABLE
    HIDDEN_STATE_VARIABLE = IMMOBILE_VARIABLE
    DIAGNOSTICS = (
        "diagnostics.two_region_exchange.mass_transfer_rate",
        "diagnostics.two_region_exchange.storage_dissipation",
        "diagnostics.two_region_exchange.memory_residual",
    )
    PARAMETERS = {
        "two_region_mass_transfer_coefficient": "First-order mass transfer coefficient alpha [s-1]",
        "two_region_mobile_water_content": "Mobile volumetric water content theta_m [m3 m-3]",
        "two_region_immobile_water_content": "Immobile volumetric water content theta_im [m3 m-3]",
    }
    memory_category = "memory"
    memory_kernel_form = "exponential"

    # Explicit config overrides precede the ParameterManager (NamespaceDefaultMixin).
    parameter_lookup_source_order = ("config", "manager")

    def __init__(
        self,
        name: str = "two_region_mass_transfer",
        *,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Two-region config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        parameter_overrides = config_dict.pop("parameters", None)
        if parameter_overrides is None:
            parameter_overrides = {}
        if not isinstance(parameter_overrides, Mapping):
            raise ValueError("Two-region parameter overrides must be a mapping.")
        device = kwargs.pop("device", None)

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Two-region mobile-immobile first-order solute exchange with exact "
                "exponential update and passive memory realization"
            ),
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "theta_m dc_m/dt = -alpha (c_m - c_im); theta_im dc_im/dt = alpha (c_m - c_im); "
                "d(t+dt) = d(t) exp(-kappa dt)"
            ),
            references=[
                "van Genuchten, M.Th. & Wierenga, P.J. (1976). Soil Sci. Soc. Am. J. 40, 473-480",
                "Haggerty, R. & Gorelick, S.M. (1995). Water Resour. Res. 31, 2383-2400",
            ],
            parameters={
                **{
                    f"{self.parameter_namespace}.{name}": text
                    for name, text in self.PARAMETERS.items()
                },
            },
            input_variables=list(self.STATE_VARIABLES),
            output_variables=[*self.STATE_VARIABLES, *self.DIAGNOSTICS],
            conservation_laws=["mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            coupling_interfaces=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            # References: van Genuchten and Wierenga (1976); Haggerty and Gorelick
            #   (1995); Hairer, Lubich and Wanner (2006), Sect. II.4-II.5.
            # Mathematical expression: Phi_h c = exp(h A) c, with
            #   A = [[-alpha/theta_m, alpha/theta_m],
            #        [alpha/theta_im, -alpha/theta_im]] and
            #   Phi_h2 Phi_h1 = Phi_(h1+h2); the implemented scalar exponential
            #   is this exact linear sub-flow, up to floating-point rounding.
            # Parameter meanings: c = (c_m, c_im) is the concentration pair;
            #   alpha is the exchange coefficient; theta_m and theta_im are
            #   mobile and immobile water contents; h is one call's duration.
            # Units: c [kg m-3]; alpha and A [s-1]; theta [m3 m-3]; h [s].
            # Domain/constraints: finite nonnegative c; positive alpha, theta
            #   and h; coefficients remain constant over the call. EXACT refers
            #   to state advancement, not time integrals of the start-point
            #   diagnostic rates. T11 checks the governed dt ladder against an
            #   independent matrix exponential and the semigroup law.
            # Sign convention: positive alpha (c_m - c_im) transfers solute
            #   from mobile to immobile; theta_m c_m + theta_im c_im is conserved.
            sub_flow_accuracy=SubFlowAccuracy.EXACT,
        )
        unknown = sorted(set(parameter_overrides).difference(self.PARAMETERS))
        if unknown:
            raise ValueError("Unknown two-region parameter override(s): " + ", ".join(unknown))
        normalized_overrides: Dict[str, Any] = {
            str(key): (dict(value) if isinstance(value, Mapping) else {"value": value})
            for key, value in parameter_overrides.items()
        }
        # config["dependencies"] seeds the instance dependencies through FunctorBase
        # (the process itself declares no static Functor dependency).
        clean_config: Dict[str, Any] = {
            "parameters": normalized_overrides,
            "dependencies": config_dict.get("dependencies", []),
        }
        transport_kwargs = {
            "device": device if device is not None else torch.device("cpu"),
        }
        super().__init__(name=name, config=clean_config, metadata=metadata, **transport_kwargs, **kwargs)
        self.optional_input_variables = []
        for parameter_name in self.PARAMETERS:
            self.get_parameter_value(parameter_name, required=True)
        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # Scientific process
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Declared IPHS structure (level 3 audit hooks; no runtime replacement)
    # ------------------------------------------------------------------
    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the theta-weighted storage functional and its effort."""

        return {
            "storage": "H = 0.5 (theta_m c_m^2 + theta_im c_im^2) [kg2 m-6]",
            "effort": "e = dH/dx = (c_m, c_im) with x = (theta_m c_m, theta_im c_im)",
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the nonnegative norm dissipation of the exchange."""

        return {
            "exchange_dissipation": "sigma = alpha (c_m - c_im)^2 >= 0 [kg2 m-6 s-1]",
            "storage_balance": "dH/dt = -sigma (closed two-region ledger)",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection; the exchange is purely dissipative."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the declared dissipation operator of the two-region exchange.

        References:
          - van Genuchten and Wierenga (1976); van der Schaft and Maschke (1995).
        Mathematical expression:
          x_dot = -R e, R = alpha [[1, -1], [-1, 1]] >= 0, J = 0.
        Parameter meanings:
          alpha mass transfer coefficient; e = (c_m, c_im) the effort pair.
        Units: R s-1 on effort kg m-3 giving kg m-3 s-1 storage rates.
        Domain/constraints: alpha > 0 makes R symmetric positive semidefinite
          with kernel (1, 1), the conserved theta-weighted total.
        Sign convention: -R e moves mass from the higher to the lower
          concentration.
        """

        prototype = state.get(self.MOBILE_VARIABLE)
        dtype = prototype.dtype if isinstance(prototype, Tensor) else torch.float64
        alpha = self.get_parameter_value("two_region_mass_transfer_coefficient", required=True)
        alpha = torch.as_tensor(alpha, dtype=dtype).reshape(())
        dissipation = alpha * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype)
        return {
            "state_vars": self.STATE_VARIABLES,
            "J": torch.zeros_like(dissipation),
            "R": dissipation,
            "ports": (),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Report the theta-weighted two-region ledger residual of one step.

        References: van Genuchten and Wierenga (1976); RELIC conservation
          architecture (Tier 1 local hook).
        Mathematical expression:
          violation = max |theta_m Delta c_m + theta_im Delta c_im|, the
          declared memory residual route of the emitted step.
        Parameter meanings: input_state and output_updates are the FunctorBase
          conservation API; the residual is the emitted diagnostic.
        Units: kg m-3.
        Domain/constraints: a missing residual diagnostic counts as zero only
          when no state update was emitted.
        Sign convention: magnitude only.
        """
        residual = output_updates.get(self.DIAGNOSTICS[2])
        if residual is None:
            if any(name in output_updates for name in self.STATE_VARIABLES):
                raise ValueError("Two-region updates were emitted without the memory residual diagnostic.")
            return {"mass": 0.0}
        value = residual.value if isinstance(residual, UpdateItem) else residual
        return {"mass": float(torch.max(torch.abs(torch.as_tensor(value))).detach())}

    def _coefficients(self, prototype: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        alpha = self.get_parameter_value("two_region_mass_transfer_coefficient", required=True).to(prototype)
        theta_m = self.get_parameter_value("two_region_mobile_water_content", required=True).to(prototype)
        theta_im = self.get_parameter_value("two_region_immobile_water_content", required=True).to(prototype)
        for label, value in (("alpha", alpha), ("theta_m", theta_m), ("theta_im", theta_im)):
            if value.numel() != 1:
                raise ValueError(f"Two-region parameter {label} must be scalar.")
            if not float(value.detach()) > 0.0:
                raise ValueError(f"Two-region parameter {label} must be positive.")
        return alpha.reshape(()), theta_m.reshape(()), theta_im.reshape(())

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """Advance the two-region exchange by the exact solution of one step.

        References:
          - van Genuchten and Wierenga (1976); Haggerty and Gorelick (1995).
        Mathematical expression:
          m = theta_m c_m + theta_im c_im; d = c_m - c_im;
          d' = d exp(-kappa dt); c_m' = (m + theta_im d')/(theta_m + theta_im);
          c_im' = (m - theta_m d')/(theta_m + theta_im);
          rate = alpha d; dissipation = -alpha d^2;
          residual = theta_m (c_m' - c_m) + theta_im (c_im' - c_im).
        Parameter meanings:
          as in the class docstring; kappa = alpha (theta_m + theta_im)/(theta_m theta_im).
        Units:
          c kg m-3; rate kg m-3 s-1; dissipation kg2 m-6 s-1; residual kg m-3.
        Domain/constraints:
          both states finite nonnegative tensors of one shape; finite positive
          dt; the update is exact for constant coefficients within the step.
        Sign convention:
          rate positive from mobile to immobile; increments sum to zero in the
          theta-weighted ledger.
        """
        # References: van Genuchten and Wierenga (1976); Coats and Smith (1964);
        #   Haggerty and Gorelick (1995) memory-function form.
        # Mathematical expression: theta_im dc_im/dt = alpha (c_m - c_im),
        #   theta_m dc_m/dt = -alpha (c_m - c_im); exact exponential update with
        #   omega = alpha (theta_m + theta_im) / (theta_m theta_im).
        # Parameter meanings: c_m mobile and c_im immobile concentrations, alpha
        #   exchange rate, theta_m and theta_im water contents.
        # Units: c kg m-3; alpha s-1; theta 1; dt s.
        # Domain/constraints: alpha >= 0; theta_m, theta_im > 0; finite positive dt;
        #   total mass theta_m c_m + theta_im c_im is conserved exactly.
        # Sign convention: positive transfer moves mass from mobile to immobile.

        if kwargs:
            raise TypeError(f"Unsupported two-region forward arguments: {sorted(kwargs)}")
        for name in self.STATE_VARIABLES:
            if name not in state:
                raise ValueError(f"{name} is required for the two-region exchange")
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"State '{name}' must be a torch.Tensor")
            if not value.is_floating_point():
                raise ValueError(f"State '{name}' must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"State '{name}' must contain only finite values")
            if bool((value < 0).any()):
                raise ValueError(f"State '{name}' must be nonnegative")
        mobile = state[self.MOBILE_VARIABLE]
        immobile = state[self.IMMOBILE_VARIABLE]
        if mobile.shape != immobile.shape:
            raise ValueError("Mobile and immobile concentrations must share one shape")
        dt_value = float(dt)
        if not torch.isfinite(mobile.new_tensor(dt_value)) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")

        alpha, theta_m, theta_im = self._coefficients(mobile)
        # References: van Genuchten and Wierenga (1976), linear two-region exchange.
        # Mathematical expression: exact exponential decay of d = c_m - c_im at
        #   rate kappa with the theta-weighted total m held fixed.
        # Parameter meanings: alpha, theta_m, theta_im as declared.
        # Units: kappa s-1; m kg m-3 (theta-weighted); d kg m-3.
        # Domain/constraints: kappa dt >= 0; exp is evaluated once per step.
        # Sign convention: d decays toward zero without changing sign.
        total = theta_m * mobile + theta_im * immobile
        difference = mobile - immobile
        kappa = alpha * (theta_m + theta_im) / (theta_m * theta_im)
        decayed = difference * torch.exp(-kappa * dt_value)
        denominator = theta_m + theta_im
        mobile_next = (total + theta_im * decayed) / denominator
        immobile_next = (total - theta_m * decayed) / denominator
        mobile_increment = mobile_next - mobile
        immobile_increment = immobile_next - immobile

        transfer_rate = (alpha * difference).detach()
        storage_dissipation = (-alpha * difference * difference).detach()
        memory_residual = (theta_m * mobile_increment + theta_im * immobile_increment).detach()

        output: FluxDict = {}
        for key, value in (
            (self.MOBILE_VARIABLE, mobile_increment),
            (self.IMMOBILE_VARIABLE, immobile_increment),
        ):
            output[key] = UpdateItem(
                value=value,
                mode=UpdateMode.ACCUMULATE,
                unit="kg m-3",
                per_timestep=True,
                source_functor=self.name,
            )
        for key, value, unit in (
            (self.DIAGNOSTICS[0], transfer_rate, "kg m-3 s-1"),
            (self.DIAGNOSTICS[1], storage_dissipation, "kg2 m-6 s-1"),
            (self.DIAGNOSTICS[2], memory_residual, "kg m-3"),
        ):
            output[key] = UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=True,
                source_functor=self.name,
            )
        return output


__all__ = ["TwoRegionMassTransferFunctor"]
