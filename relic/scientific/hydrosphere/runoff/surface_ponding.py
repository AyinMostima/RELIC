"""Hortonian surface ponding as a hybrid topology transition (G09-TOPO).

References:
    - Horton, R.E. (1933). The role of infiltration in the hydrologic cycle.
      Trans. Am. Geophys. Union 14, 446-460 (infiltration excess).
    - Beven, K.J. (2012). Rainfall-Runoff Modelling: The Primer, 2nd ed.,
      Wiley, ch. 6 (depression storage and runoff generation thresholds).
    - Dooge, J.C.I. (1973). Linear Theory of Hydrologic Systems, USDA Tech.
      Bull. 1468 (linear reservoir recession).
    - Goebel, R., Sanfelice, R.G. and Teel, A.R. (2012). Hybrid Dynamical
      Systems, Princeton (event-triggered jumps and topology resets).
    - Revised RELIC foundations, section E: hybrid topology transition
      x_h^+ = P_h x_h^- + R_h eta_h with a ledger audit.

Two Functors realize one real connectivity change of the process graph:

``HortonianPondingEventFunctor`` (PhysicalField contract) is always
connected. It converts the rainfall depth increment of the step into the
infiltration excess above the surface capacity and, at the accepted-step
boundary, requests the topology transition that connects
(``ponding_onset``) or retires (``ponding_cessation``) the ponding store.
The request is an ``ExternalIntent`` executed by the simulator lifecycle
owner after the step is accepted; the step itself runs entirely on the
committed provider view it started with.

``SurfacePondingFunctor`` (TransportMechanism contract) exists in the graph
only between onset and cessation. It stores the ponded depth and drains it
as overland flow with an exact linear-reservoir recession.

The topology remaps are declared by :func:`ponding_onset_transition` and
:func:`ponding_cessation_transition` on the state triple
(ponded depth, excess, connectivity flag) and are committed by the
registration owner with a single epoch publication and a ledger audit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor

from ....core.abstractions import FluxDict, StateDict
from ....core.functor.functor_bridge_utils import NamespaceDefaultMixin
from ....core.functor import (
    DependencyList,
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    TransportMechanismFunctor,
    VariableKind,
)
from ....core.state import ExternalIntent, UpdateItem, UpdateMode
from ....utils.time_constants import SECONDS_PER_HOUR

RAINFALL_DEPTH_VARIABLE = "stochastic_rainfall_depth"
PONDED_DEPTH_VARIABLE = "surface_ponded_depth"
EXCESS_VARIABLE = "surface_ponding_excess"
ACTIVE_VARIABLE = "surface_ponding_active"
# Parameter namespace of both ponding functors; passed explicitly on every
# parameter read so the literal keys resolve to the hydrosphere.runoff.*
# entries of the parameter database.
PARAMETER_NAMESPACE = "hydrosphere.runoff"
TRANSITION_ONSET = "ponding_onset"
TRANSITION_CESSATION = "ponding_cessation"
# Output keys carrying the topology intents; an intent is a delivery request,
# never a scientific output, so these keys are not metadata output_variables.
ONSET_INTENT_OUTPUT = "topology_intent_ponding_onset"
CESSATION_INTENT_OUTPUT = "topology_intent_ponding_cessation"
TOPOLOGY_INTENT_OPERATION = "topology"


def _require_finite_float_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"State '{name}' must be a torch.Tensor")
    if not value.is_floating_point():
        raise ValueError(f"State '{name}' must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"State '{name}' must contain only finite values")
    return value


def _require_positive_dt(dt: Any, like: Tensor) -> float:
    dt_value = float(dt)
    if not torch.isfinite(like.new_tensor(dt_value)) or dt_value <= 0.0:
        raise ValueError("dt must be finite and positive")
    return dt_value


def _normalize_overrides(overrides: Any, allowed: Mapping[str, str], label: str) -> Dict[str, Any]:
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{label} parameter overrides must be a mapping.")
    unknown = sorted(set(overrides).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {label} parameter override(s): " + ", ".join(unknown))
    return {
        str(key): (dict(value) if isinstance(value, Mapping) else {"value": value})
        for key, value in overrides.items()
    }


class HortonianPondingEventFunctor(NamespaceDefaultMixin, PhysicalFieldFunctor):
    """Infiltration-excess detector that requests the ponding topology transition.

    Mathematical expression:
      Delta H_n = H_n - H_{n-1} (cumulative rainfall depth increment);
      E_n = max(0, Delta H_n - I_max dt / 3600);
      onset request when a = 0 and max_cells E_n >= E_on;
      cessation request when a = 1, E_n = 0 and max_cells h_n <= h_off.
    Parameter meanings:
      H cumulative rainfall depth, I_max infiltration capacity, E excess,
      E_on onset threshold, h ponded depth, h_off cessation threshold,
      a connectivity flag (0/1, uniform over cells).
    Units:
      H, Delta H, E, E_on, h, h_off mm; I_max mm h-1; dt s; a 1.
    Domain/constraints:
      finite floating-point tensors of one shape; Delta H >= 0 (a decreasing
      cumulative depth is rejected); a in {0, 1} and uniform over cells; one
      request per boundary and per transition.
    Sign convention:
      excess is the nonnegative part of the rainfall increment above the
      infiltration capacity; infiltrated excess is reported only while the
      store is disconnected.
    """

    parameter_namespace = PARAMETER_NAMESPACE
    topology_category = "topology_change"
    STATE_VARIABLES = (EXCESS_VARIABLE,)
    DIAGNOSTICS = (
        "diagnostics.ponding_event.excess_depth",
        "diagnostics.ponding_event.rainfall_increment",
        "diagnostics.ponding_event.infiltrated_excess",
    )
    EXTRA_STATE_FORMAT = "relic.ponding_event.rainfall_memory"
    EXTRA_STATE_VERSION = 1
    PARAMETERS = {
        "ponding_infiltration_capacity": "Steady surface infiltration capacity I_max [mm h-1]",
        "ponding_onset_depth": "Excess depth connecting the ponding store [mm]",
        "ponding_cessation_depth": "Ponded depth retiring the ponding store [mm]",
    }

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    # Explicit config overrides precede the ParameterManager (NamespaceDefaultMixin).
    parameter_lookup_source_order = ("config", "manager")

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "ponding_event",
        *,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Ponding event config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        overrides = _normalize_overrides(config_dict.pop("parameters", None), self.PARAMETERS, "ponding event")
        rainfall_provider = config_dict.pop("rainfall_provider", "stochastic_rainfall")
        if not isinstance(rainfall_provider, str) or not rainfall_provider.strip():
            raise ValueError("rainfall_provider must be a non-empty Functor name.")
        device = kwargs.pop("device", None)
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Hortonian infiltration-excess detector requesting the surface "
                "ponding topology transition at the accepted-step boundary"
            ),
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "E_n = max(0, H_n - H_{n-1} - I_max dt / 3600); "
                "onset: a = 0 and max E_n >= E_on; "
                "cessation: a = 1, E_n = 0 and max h_n <= h_off"
            ),
            references=[
                "Horton, R.E. (1933). Trans. Am. Geophys. Union 14, 446-460",
                "Beven, K.J. (2012). Rainfall-Runoff Modelling: The Primer, 2nd ed., Wiley",
                "Goebel, R., Sanfelice, R.G. & Teel, A.R. (2012). Hybrid Dynamical Systems, Princeton",
            ],
            parameters={
                f"{self.parameter_namespace}.{short_name}": description
                for short_name, description in self.PARAMETERS.items()
            },
            input_variables=[RAINFALL_DEPTH_VARIABLE, ACTIVE_VARIABLE, PONDED_DEPTH_VARIABLE],
            output_variables=[EXCESS_VARIABLE, *self.DIAGNOSTICS],
            conservation_laws=["mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            coupling_interfaces=["precipitation_flux"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        clean_config: Dict[str, Any] = {
            "parameters": overrides,
            # Instance names from config["dependencies"]; the rainfall provider
            # is the static declaration merged by get_dependencies.
            "dependencies": config_dict.get("dependencies", []),
            "rainfall_provider": rainfall_provider,
        }
        super().__init__(name=name, config=clean_config, metadata=metadata, **kwargs)
        self.optional_input_variables = []
        self._rainfall_provider = rainfall_provider
        # Replay-critical hidden state: the cumulative rainfall depth seen at
        # the previous accepted step (module extra state, F02 contract).
        self._last_rainfall_depth: Optional[Tensor] = None
        for parameter_name in self.PARAMETERS:
            self.get_parameter_value(parameter_name, required=True)
        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------
    def get_dependencies(self) -> DependencyList:
        return self._merge_declared_dependencies([self._rainfall_provider])

    def get_extra_state(self) -> Dict[str, Any]:
        last = self._last_rainfall_depth
        return {
            "format": self.EXTRA_STATE_FORMAT,
            "version": self.EXTRA_STATE_VERSION,
            "last_rainfall_depth": None if last is None else last.detach().cpu().clone(),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping) or set(state) != {"format", "version", "last_rainfall_depth"}:
            raise ValueError("Ponding event extra state keys do not match the schema.")
        if state["format"] != self.EXTRA_STATE_FORMAT or state["version"] != self.EXTRA_STATE_VERSION:
            raise ValueError("Ponding event extra state format or version is unsupported.")
        last = state["last_rainfall_depth"]
        if last is None:
            self._last_rainfall_depth = None
            return
        if not isinstance(last, Tensor) or not last.is_floating_point():
            raise TypeError("Ponding event last_rainfall_depth must be a floating-point tensor.")
        self._last_rainfall_depth = last.detach().clone().to(device=self.device)

    @property
    def last_rainfall_depth(self) -> Optional[Tensor]:
        return None if self._last_rainfall_depth is None else self._last_rainfall_depth.clone()

    # ------------------------------------------------------------------
    # Scientific process
    # ------------------------------------------------------------------
    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References: Horton (1933) Trans. AGU 14, 446-460; Beven (2012)
        #   Rainfall-Runoff Modelling, ch. 5 (infiltration-excess overland flow).
        # Mathematical expression: dH = H(t) - H(t - dt); E = max(0, dH - I_max dt);
        #   onset when a = 0 and max(E) >= E_on; cessation when a = 1, E = 0 and
        #   max(h) <= h_off.
        # Parameter meanings: H cumulative rainfall depth, I_max infiltration
        #   capacity, E infiltration-excess depth, a ponding connectivity flag,
        #   h ponded depth, E_on onset threshold, h_off cessation threshold.
        # Units: H, E, h, E_on, h_off in mm; I_max in mm s-1; dt in s; a in 1.
        # Domain/constraints: dH >= 0 (a decreasing cumulative depth is rejected);
        #   a in {0, 1} and uniform over cells; dt > 0; one request per boundary.
        # Sign convention: E is the nonnegative excess above the infiltration
        #   capacity; infiltrated excess is reported only while disconnected.
        if kwargs:
            raise TypeError(f"Unsupported ponding event forward arguments: {sorted(kwargs)}")
        for name in (RAINFALL_DEPTH_VARIABLE, ACTIVE_VARIABLE, PONDED_DEPTH_VARIABLE):
            if name not in state:
                raise ValueError(f"{name} is required by the ponding event detector")
        depth = _require_finite_float_tensor(state[RAINFALL_DEPTH_VARIABLE], RAINFALL_DEPTH_VARIABLE)
        active = _require_finite_float_tensor(state[ACTIVE_VARIABLE], ACTIVE_VARIABLE)
        ponded = _require_finite_float_tensor(state[PONDED_DEPTH_VARIABLE], PONDED_DEPTH_VARIABLE)
        if active.shape != depth.shape or ponded.shape != depth.shape:
            raise ValueError("Ponding event inputs must share one shape")
        dt_value = _require_positive_dt(dt, depth)
        active_values = active.detach().flatten()
        if active_values.numel() == 0:
            raise ValueError("Ponding event requires at least one cell")
        connected = bool(active_values[0] == 1.0)
        if not bool(((active_values == 0.0) | (active_values == 1.0)).all()) or not bool(
            (active_values == active_values[0]).all()
        ):
            raise ValueError("surface_ponding_active must be 0 or 1 and uniform over cells")

        capacity = self.get_parameter_value(
            "ponding_infiltration_capacity", namespace=PARAMETER_NAMESPACE, required=True
        ).to(depth)
        onset_depth = self.get_parameter_value(
            "ponding_onset_depth", namespace=PARAMETER_NAMESPACE, required=True
        ).to(depth)
        cessation_depth = self.get_parameter_value(
            "ponding_cessation_depth", namespace=PARAMETER_NAMESPACE, required=True
        ).to(depth)
        if capacity.numel() != 1 or onset_depth.numel() != 1 or cessation_depth.numel() != 1:
            raise ValueError("Ponding event parameters must be scalar.")

        # References: Horton (1933); Beven (2012) ch. 6.
        # Mathematical expression: Delta H = H_n - H_{n-1};
        #   E = max(0, Delta H - I_max dt / 3600).
        # Parameter meanings: H cumulative rainfall depth; I_max infiltration
        #   capacity; E infiltration excess of the step.
        # Units: H, Delta H, E mm; I_max mm h-1; dt s; 3600 s h-1.
        # Domain/constraints: Delta H >= 0; first call after initialization
        #   has no previous depth and emits zero excess.
        # Sign convention: E >= 0 is water that cannot infiltrate in the step.
        if self._last_rainfall_depth is None:
            increment = torch.zeros_like(depth)
        else:
            previous = self._last_rainfall_depth.to(device=depth.device, dtype=depth.dtype)
            if previous.shape != depth.shape:
                raise ValueError("Cumulative rainfall depth changed shape between steps")
            increment = depth - previous
            if bool((increment < 0.0).any()):
                raise ValueError("Cumulative rainfall depth decreased between steps")
        self._last_rainfall_depth = depth.detach().clone()
        capacity_depth = capacity.reshape(()) * depth.new_tensor(dt_value / float(SECONDS_PER_HOUR))
        excess = torch.clamp(increment - capacity_depth, min=0.0)
        infiltrated_excess = excess * (0.0 if connected else 1.0)

        outputs: FluxDict = {
            EXCESS_VARIABLE: UpdateItem(
                value=excess,
                mode=UpdateMode.REPLACE,
                unit="mm",
                per_timestep=False,
                source_functor=self.name,
            ),
            self.DIAGNOSTICS[0]: excess.detach(),
            self.DIAGNOSTICS[1]: increment.detach(),
            self.DIAGNOSTICS[2]: infiltrated_excess.detach(),
        }
        # References: Goebel, Sanfelice and Teel (2012), event-triggered jumps.
        # Mathematical expression: onset guard a = 0 and max E >= E_on;
        #   cessation guard a = 1, E = 0 and max h <= h_off.
        # Parameter meanings: E_on onset threshold, h_off cessation threshold.
        # Units: mm.
        # Domain/constraints: one request per transition and boundary; the
        #   request is a pure descriptor executed after step acceptance.
        # Sign convention: not applicable.
        time_tag = f"{float(time):.6f}"
        if not connected and bool((excess.detach() >= onset_depth.reshape(())).any()):
            outputs[ONSET_INTENT_OUTPUT] = ExternalIntent(
                TOPOLOGY_INTENT_OPERATION,
                TRANSITION_ONSET,
                f"{self.name}:{TRANSITION_ONSET}:{time_tag}",
                {"transition": TRANSITION_ONSET},
            )
        elif (
            connected
            and bool((excess.detach() == 0.0).all())
            and bool((ponded.detach() <= cessation_depth.reshape(())).all())
        ):
            outputs[CESSATION_INTENT_OUTPUT] = ExternalIntent(
                TOPOLOGY_INTENT_OPERATION,
                TRANSITION_CESSATION,
                f"{self.name}:{TRANSITION_CESSATION}:{time_tag}",
                {"transition": TRANSITION_CESSATION},
            )
        return outputs

    def check_conservation_laws(self, input_state: StateDict, output_updates: FluxDict) -> Dict[str, float]:
        """The rainfall increment is partitioned exactly into infiltration and excess.

        References: Horton (1933).
        Mathematical expression: residual = Delta H - min(Delta H, I_max dt) - E = 0.
        Parameter meanings: as in forward.
        Units: mm.
        Domain/constraints: reported from the emitted diagnostics of the step.
        Sign convention: magnitude only.
        """
        excess = output_updates.get(self.DIAGNOSTICS[0])
        increment = output_updates.get(self.DIAGNOSTICS[1])
        if excess is None or increment is None:
            if EXCESS_VARIABLE in output_updates:
                raise ValueError("Ponding event updates were emitted without their diagnostics.")
            return {"mass": 0.0}
        excess_value = torch.as_tensor(excess.value if isinstance(excess, UpdateItem) else excess)
        increment_value = torch.as_tensor(increment.value if isinstance(increment, UpdateItem) else increment)
        infiltration = torch.minimum(increment_value, increment_value - excess_value)
        residual = increment_value - infiltration - excess_value
        return {"mass": float(torch.max(torch.abs(residual)).detach())}


class SurfacePondingFunctor(NamespaceDefaultMixin, TransportMechanismFunctor):
    """Ponded surface water draining as overland flow (linear reservoir).

    Mathematical expression:
      h_0 = h_n + E_n (excess enters the store at the start of the step);
      h_{n+1} = h_0 exp(-dt / tau); q_n = h_0 - h_{n+1};
      Delta h_n = h_{n+1} - h_n.
    Parameter meanings:
      h ponded depth, E infiltration excess of the step, tau linear-reservoir
      residence time, q overland-flow depth drained in the step.
    Units:
      h, E, q, Delta h mm; tau, dt s.
    Domain/constraints:
      finite nonnegative tensors of one shape; tau > 0; finite positive dt;
      exact recession (no time-step restriction).
    Sign convention:
      q >= 0 leaves the store; Delta h = E - q.
    """

    TRANSPORT_TYPE = "advection"

    parameter_namespace = "hydrosphere.runoff"
    STATE_VARIABLES = (PONDED_DEPTH_VARIABLE,)
    DIAGNOSTICS = (
        "diagnostics.surface_ponding.runoff_depth",
        "diagnostics.surface_ponding.water_balance_residual",
    )
    PARAMETERS = {"ponding_drainage_time": "Linear-reservoir residence time tau [s]"}

    # Explicit config overrides precede the ParameterManager (NamespaceDefaultMixin).
    parameter_lookup_source_order = ("config", "manager")

    def __init__(
        self,
        name: str = "surface_ponding",
        *,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Surface ponding config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        overrides = _normalize_overrides(config_dict.pop("parameters", None), self.PARAMETERS, "surface ponding")
        event_provider = config_dict.pop("event_provider", "ponding_event")
        if not isinstance(event_provider, str) or not event_provider.strip():
            raise ValueError("event_provider must be a non-empty Functor name.")
        device = kwargs.pop("device", None)
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Surface ponded water store fed by Hortonian excess and drained "
                "as overland flow with an exact linear-reservoir recession"
            ),
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "h_0 = h_n + E_n; h_{n+1} = h_0 exp(-dt/tau); q_n = h_0 - h_{n+1}"
            ),
            references=[
                "Dooge, J.C.I. (1973). Linear Theory of Hydrologic Systems, USDA Tech. Bull. 1468",
                "Beven, K.J. (2012). Rainfall-Runoff Modelling: The Primer, 2nd ed., Wiley",
            ],
            parameters={
                f"{self.parameter_namespace}.{short_name}": description
                for short_name, description in self.PARAMETERS.items()
            },
            input_variables=[PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE],
            output_variables=[PONDED_DEPTH_VARIABLE, *self.DIAGNOSTICS],
            conservation_laws=["mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            coupling_interfaces=["channel_routing"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        clean_config: Dict[str, Any] = {
            "parameters": overrides,
            # Instance names from config["dependencies"]; the event provider
            # is the static declaration merged by get_dependencies.
            "dependencies": config_dict.get("dependencies", []),
            "event_provider": event_provider,
        }
        super().__init__(name=name, config=clean_config, metadata=metadata, **kwargs)
        self.optional_input_variables = []
        self._event_provider = event_provider
        for parameter_name in self.PARAMETERS:
            self.get_parameter_value(parameter_name, required=True)
        if device is not None:
            self.to_device(device)

    def get_dependencies(self) -> DependencyList:
        return self._merge_declared_dependencies([self._event_provider])

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References: Dooge (1973) Linear Theory of Hydrologic Systems, USDA
        #   Tech. Bull. 1468 (linear reservoir); Beven (2012) ch. 5.
        # Mathematical expression: h(t + dt) = (h + E) exp(-dt / tau);
        #   runoff depth R = (h + E) (1 - exp(-dt / tau)); residual
        #   r = (h + E) - h(t + dt) - R.
        # Parameter meanings: h ponded depth, E infiltration-excess input of the
        #   step, tau drainage time scale, R drained (overland-flow) depth.
        # Units: h, E, R, r in mm; tau and dt in s.
        # Domain/constraints: h >= 0, E >= 0, tau > 0 scalar, dt > 0; the store
        #   exists only while the ponding connectivity flag is 1.
        # Sign convention: the state increment is h(t + dt) - h (negative when
        #   drainage exceeds input); R is positive out of the store.
        if kwargs:
            raise TypeError(f"Unsupported surface ponding forward arguments: {sorted(kwargs)}")
        for name in (PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE):
            if name not in state:
                raise ValueError(f"{name} is required by the surface ponding store")
        ponded = _require_finite_float_tensor(state[PONDED_DEPTH_VARIABLE], PONDED_DEPTH_VARIABLE)
        excess = _require_finite_float_tensor(state[EXCESS_VARIABLE], EXCESS_VARIABLE)
        if excess.shape != ponded.shape:
            raise ValueError("Surface ponding inputs must share one shape")
        if bool((ponded < 0.0).any()) or bool((excess < 0.0).any()):
            raise ValueError("Ponded depth and excess must be nonnegative")
        dt_value = _require_positive_dt(dt, ponded)
        tau = self.get_parameter_value(
            "ponding_drainage_time", namespace=PARAMETER_NAMESPACE, required=True
        ).to(ponded)
        if tau.numel() != 1 or not bool(tau.reshape(()) > 0.0):
            raise ValueError("ponding_drainage_time must be a positive scalar")

        # References: Dooge (1973), linear reservoir; Beven (2012).
        # Mathematical expression: h_0 = h + E; h' = h_0 exp(-dt/tau);
        #   q = h_0 - h'; Delta h = h' - h.
        # Parameter meanings: h ponded depth, E excess inflow, tau residence
        #   time, q drained overland-flow depth.
        # Units: h, E, q mm; tau, dt s.
        # Domain/constraints: exact solution of dh/dt = -h/tau over the step.
        # Sign convention: q >= 0 leaves the store.
        start = ponded + excess
        decay = torch.exp(-ponded.new_tensor(dt_value) / tau.reshape(()))
        end = start * decay
        runoff = start - end
        delta = end - ponded
        residual = delta - excess + runoff
        return {
            PONDED_DEPTH_VARIABLE: UpdateItem(
                value=delta,
                mode=UpdateMode.ACCUMULATE,
                unit="mm",
                per_timestep=True,
                source_functor=self.name,
            ),
            self.DIAGNOSTICS[0]: runoff.detach(),
            self.DIAGNOSTICS[1]: residual.detach(),
        }

    def check_conservation_laws(self, input_state: StateDict, output_updates: FluxDict) -> Dict[str, float]:
        """Report the ponded-water budget residual of the emitted step.

        References: Dooge (1973).
        Mathematical expression: residual = Delta h - E + q.
        Parameter meanings: as in forward.
        Units: mm.
        Domain/constraints: reported from the emitted residual diagnostic.
        Sign convention: magnitude only.
        """
        residual = output_updates.get(self.DIAGNOSTICS[1])
        if residual is None:
            if PONDED_DEPTH_VARIABLE in output_updates:
                raise ValueError("Surface ponding updates were emitted without the budget residual.")
            return {"mass": 0.0}
        value = residual.value if isinstance(residual, UpdateItem) else residual
        return {"mass": float(torch.max(torch.abs(torch.as_tensor(value))).detach())}


# ----------------------------------------------------------------------
# Declared topology transitions (consumed by the registration owner)
# ----------------------------------------------------------------------
def ponding_onset_transition(
    ponding_functor: SurfacePondingFunctor,
    *,
    execution_group: str,
) -> Any:
    """Connect the ponding store: x^+ = P x^- + R eta on (h, E, a).

    References: Goebel, Sanfelice and Teel (2012); revised RELIC foundations E.
    Mathematical expression:
      h^+ = h^- + E^-, E^+ = 0, a^+ = 1, i.e.
      P = [[1, 1, 0], [0, 0, 0], [0, 0, 0]], R eta = (0, 0, 1);
      ledger C = [1, 1, 0] (water depth): C x^+ = C x^-.
    Parameter meanings: h ponded depth, E excess, a connectivity flag.
    Units: h, E mm; a 1.
    Domain/constraints: guard a^- = 0; the store is not yet registered.
    Sign convention: excess water moves into the ponded store; nothing leaves.
    """
    from ....core.simulator.registration import RemapRow, TopologyTransition

    return TopologyTransition(
        name=TRANSITION_ONSET,
        event_name=TRANSITION_ONSET,
        retire=(),
        connect=((ponding_functor, execution_group),),
        state_remap=(
            RemapRow(target=PONDED_DEPTH_VARIABLE, sources={PONDED_DEPTH_VARIABLE: 1.0, EXCESS_VARIABLE: 1.0}),
            RemapRow(target=EXCESS_VARIABLE, sources={}),
            RemapRow(target=ACTIVE_VARIABLE, sources={}, offset=1.0),
        ),
        ledger_variables=(PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE),
        ledger_change_source=None,
        guard=(ACTIVE_VARIABLE, 0.0),
        parameter_keys=(
            "hydrosphere.runoff.ponding_infiltration_capacity",
            "hydrosphere.runoff.ponding_onset_depth",
        ),
        audit_route="diagnostics.topology_transition.ledger_residual",
        ledger_change_route="diagnostics.topology_transition.ledger_change",
    )


def ponding_cessation_transition(
    ponding_functor_name: str = "surface_ponding",
) -> Any:
    """Retire the ponding store: residual ponded water infiltrates (declared Pi).

    References: Goebel, Sanfelice and Teel (2012); Beven (2012) ch. 6.
    Mathematical expression:
      h^+ = 0, E^+ = E^-, a^+ = 0, i.e. P = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
      declared ledger change Pi = -h^- (infiltrated), audited as
      C x^+ - C x^- - Pi = 0 with C = [1, 1, 0].
    Parameter meanings: h ponded depth at retirement, E excess, a flag.
    Units: mm; a 1.
    Domain/constraints: guard a^- = 1; the store is registered.
    Sign convention: Pi < 0 is water leaving the surface ledger by infiltration.
    """
    from ....core.simulator.registration import RemapRow, TopologyTransition

    return TopologyTransition(
        name=TRANSITION_CESSATION,
        event_name=TRANSITION_CESSATION,
        retire=(ponding_functor_name,),
        connect=(),
        state_remap=(
            RemapRow(target=PONDED_DEPTH_VARIABLE, sources={}),
            RemapRow(target=ACTIVE_VARIABLE, sources={}, offset=0.0),
        ),
        ledger_variables=(PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE),
        ledger_change_source=PONDED_DEPTH_VARIABLE,
        guard=(ACTIVE_VARIABLE, 1.0),
        parameter_keys=("hydrosphere.runoff.ponding_cessation_depth",),
        audit_route="diagnostics.topology_transition.ledger_residual",
        ledger_change_route="diagnostics.topology_transition.ledger_change",
    )


__all__: List[str] = [
    "ACTIVE_VARIABLE",
    "EXCESS_VARIABLE",
    "HortonianPondingEventFunctor",
    "PONDED_DEPTH_VARIABLE",
    "RAINFALL_DEPTH_VARIABLE",
    "SurfacePondingFunctor",
    "TRANSITION_CESSATION",
    "TRANSITION_ONSET",
    "ponding_cessation_transition",
    "ponding_onset_transition",
]
