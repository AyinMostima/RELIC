"""Marked-Poisson stochastic rainfall (marked_jump semantics).

Theory document: docs/source/theory/atmosphere_stochastic_rainfall.md

Rainfall arrivals form a homogeneous Poisson counting process N_t with rate
lambda and every arrival carries an exponential depth mark z with mean alpha
(Rodriguez-Iturbe et al. 1999; Laio et al. 2001).  Each governed step samples
the arrivals inside [t, t+dt) with the existing direct Gillespie SSA owner
(relic.stochastic.gillespie.GillespieSSA, one zero-order channel) and the
marks from the same Functor-owned substream, then emits the uncompensated
increments dN and dH = sum z_i together with their predictable compensators
lambda dt and lambda alpha dt.  The compensated representation of the same
process must add those compensators as drift; emitting N (not tilde N) keeps
the process mean unchanged (revised foundations, marked jump semantics).
The noise is an external forcing closure; no fluctuation-dissipation
relation is claimed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from ...core.abstractions import FluxDict, StateDict
from ...core.functor.functor_bridge_utils import NamespaceDefaultMixin
from ...core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    FunctorSubstreamMixin,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from ...core.state import UpdateItem, UpdateMode
from ...stochastic.gillespie import GillespieSSA, Reaction, ReactionSystem
from ...utils.time_constants import SECONDS_PER_DAY


class MarkedPoissonRainfallFunctor(FunctorSubstreamMixin, NamespaceDefaultMixin, PhysicalFieldFunctor):
    """Marked Poisson rainfall dN ~ Poisson(lambda dt), dH = sum z_i (PhysicalField contract).

    References:
      - Rodriguez-Iturbe, I., Porporato, A., Ridolfi, L., Isham, V. & Cox,
        D.R. (1999). Proc. R. Soc. A 455, 3789-3805 (Poisson rainfall with
        exponential depths).
      - Laio, F., Porporato, A., Ridolfi, L. & Rodriguez-Iturbe, I. (2001).
        Adv. Water Resour. 24, 707-723.
      - Gillespie, D.T. (1977). J. Phys. Chem. 81, 2340-2361 (direct method).
      - Karlin, S. & Taylor, H.M. (1981). A Second Course in Stochastic
        Processes, compound Poisson processes.
    Mathematical expression:
      dN_t = N(dt), N ~ Poisson(lambda dt);  dH_t = sum_{i=1}^{dN_t} z_i,
      z_i ~ Exp(mean alpha);  compensators  E[dN_t] = lambda dt,
      E[dH_t] = lambda alpha dt.
    Parameter meanings:
      N event count, H accumulated depth, lambda arrival rate, alpha mean
      depth mark, dt the simulator step converted to days.
    Units:
      N 1; H mm; lambda day-1; alpha mm; dt s -> day; compensators 1 and mm.
    Domain/constraints:
      lambda > 0 and alpha > 0 (ParameterManager bounds); finite positive dt;
      integer-valued nonnegative count state; the SSA arrival cap per step
      (mu + 12 sqrt(mu) + 64, bounded by MAX_EVENTS_PER_STEP) is a fail-closed
      guard, never a truncation.
    Sign convention:
      every arrival adds one count and one nonnegative depth; there are no
      negative marks and no removal channel.
    """

    parameter_namespace = "stochastic.weather.rainfall"
    COUNT_VARIABLE = "stochastic_rainfall_event_count"
    DEPTH_VARIABLE = "stochastic_rainfall_depth"
    STATE_VARIABLES = (COUNT_VARIABLE, DEPTH_VARIABLE)
    DIAGNOSTICS = (
        "diagnostics.stochastic_rainfall.event_count_increment",
        "diagnostics.stochastic_rainfall.depth_increment",
        "diagnostics.stochastic_rainfall.compensator_count",
        "diagnostics.stochastic_rainfall.compensator_depth",
    )
    EXTRA_STATE_FORMAT = "relic.stochastic_rainfall.event_substream"
    EXTRA_STATE_VERSION = 1
    SUBSTREAM_LABEL = "Stochastic rainfall"
    EVENT_SPECIES = "rain_event"
    ARRIVAL_REACTION = "arrival"
    # Fail-closed arrival guard per cell and step: the SSA owner preallocates
    # its trajectory buffers, so the cap is mu + ARRIVAL_CAP_SIGMAS sqrt(mu) +
    # ARRIVAL_CAP_BASE with mu = lambda dt; a Poisson variate exceeds that by
    # a Chernoff-negligible probability, and reaching it raises instead of
    # truncating.  MAX_EVENTS_PER_STEP bounds the admissible mu itself.
    ARRIVAL_CAP_BASE = 64
    ARRIVAL_CAP_SIGMAS = 12.0
    MAX_EVENTS_PER_STEP = 100000
    stochastic_category = "marked_jump"
    noise_classification = "external_forcing"
    solver_name = "direct_ssa"
    mark_law = "exponential"
    # Override whitelist and runtime short names (namespace parameter_namespace);
    # metadata.parameters declares the full ParameterManager keys.
    PARAMETERS = {
        "event_rate": "Poisson arrival rate lambda of rainfall events [day-1]",
        "mean_depth": "Mean alpha of the exponential depth mark [mm]",
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
        name: str = "stochastic_rainfall",
        *,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Stochastic rainfall config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        seed = self.validate_substream_seed(config_dict.pop("seed", None))
        parameter_overrides = config_dict.pop("parameters", None)
        if parameter_overrides is None:
            parameter_overrides = {}
        if not isinstance(parameter_overrides, Mapping):
            raise ValueError("Stochastic rainfall parameter overrides must be a mapping.")
        device = kwargs.pop("device", None)

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Marked Poisson rainfall: direct-SSA arrivals with exponential depth "
                "marks from a Functor-owned random substream"
            ),
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "dN = N(dt), N ~ Poisson(lambda dt); dH = sum_{i<=dN} z_i, z_i ~ Exp(alpha); "
                "compensators lambda dt and lambda alpha dt"
            ),
            references=[
                "Rodriguez-Iturbe, I. et al. (1999). Proc. R. Soc. A 455, 3789-3805",
                "Laio, F. et al. (2001). Adv. Water Resour. 24, 707-723",
                "Gillespie, D.T. (1977). J. Phys. Chem. 81, 2340-2361",
                "Karlin, S. & Taylor, H.M. (1981). A Second Course in Stochastic Processes",
            ],
            parameters={
                f"{self.parameter_namespace}.{short_name}": description
                for short_name, description in self.PARAMETERS.items()
            },
            input_variables=list(self.STATE_VARIABLES),
            output_variables=[*self.STATE_VARIABLES, *self.DIAGNOSTICS],
            conservation_laws=["none"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            coupling_interfaces=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        unknown = sorted(set(parameter_overrides).difference(self.PARAMETERS))
        if unknown:
            raise ValueError("Unknown stochastic rainfall parameter override(s): " + ", ".join(unknown))
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
        super().__init__(name=name, config=clean_config, metadata=metadata, **kwargs)
        self.optional_input_variables = []
        self._initialize_substream(seed)
        self._jump_pending: Optional[Dict[str, Any]] = None
        self._trial_jump_spec: Optional[Dict[str, Any]] = None
        self._arrival_ssa: Optional[GillespieSSA] = None
        self._arrival_rate: Optional[float] = None
        for parameter_name in self.PARAMETERS:
            self.get_parameter_value(parameter_name, required=True)
        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # Marked-jump path ledger: variable steps stay on one event path
    # ------------------------------------------------------------------
    def _sample_jump_events(
        self,
        rate_value: float,
        window_days: float,
        n_cells: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """Draw arrivals and unit marks on [0, window_days) for every cell.

        References: Gillespie (1977) direct method; Rodriguez-Iturbe et al.
          (1999) exponential marks; Karlin and Taylor (1981) compound Poisson.
        Mathematical expression: per cell c, arrival times 0 < t_1 < ... <
          t_k < window with Exp(lambda) gaps; unit marks E_i ~ Exp(1) in
          (cell, time) order; depth marks alpha E_i are applied by the caller.
        Parameter meanings: lambda day-1, window in days, E_i unit marks.
        Units: times day; marks 1.
        Domain/constraints: a cell reaching the arrival cap fails closed; the
          draw order (arrivals, then marks in cell order) is fixed so a
          rewound substream replays the same events.
        Sign convention: nonnegative counts and marks.
        """
        cap = self.arrival_cap(rate_value * window_days)
        ssa = self._arrival_process(rate_value)
        sample = ssa.sample(
            torch.zeros(1, dtype=torch.int64, device=device),
            num_samples=n_cells,
            final_time=window_days,
            max_reactions=cap,
            generator=self.generator,
        )
        arrivals = sample["trajectory_lengths"] - 1
        if bool((arrivals >= cap).any()):
            raise RuntimeError(
                "Stochastic rainfall reached the per-step arrival cap; the step is "
                "rejected instead of truncating the arrival count."
            )
        total_arrivals = int(arrivals.sum())
        cell_index = torch.repeat_interleave(
            torch.arange(n_cells, device=device), arrivals
        )
        times = sample["times"].to(dtype=dtype)
        if total_arrivals > 0:
            event_time = torch.cat([
                times[cell, 1:1 + int(arrivals[cell])] for cell in range(n_cells)
            ])
            unit_marks = torch.empty(total_arrivals, dtype=dtype, device=device)
            unit_marks.exponential_(1.0, generator=self.generator)
        else:
            event_time = torch.zeros(0, dtype=dtype, device=device)
            unit_marks = torch.zeros(0, dtype=dtype, device=device)
        return {"cell_index": cell_index, "event_time": event_time, "unit_mark": unit_marks}

    @staticmethod
    def _concat_events(first: Dict[str, Tensor], second: Dict[str, Tensor], offset_days: float) -> Dict[str, Tensor]:
        return {
            "cell_index": torch.cat([first["cell_index"], second["cell_index"]]),
            "event_time": torch.cat([first["event_time"], second["event_time"] + offset_days]),
            "unit_mark": torch.cat([first["unit_mark"], second["unit_mark"]]),
        }

    def _events_on_path(
        self,
        time_days: float,
        window_days: float,
        rate_value: float,
        n_cells: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """Return the events of [time, time + window) consistent with the ledger.

        Mathematical expression: a pending event set on [t, t + H) restricted
          to [t, t + h) for h < H keeps exactly the events with time < h and
          leaves the rest (shifted by -h) pending at t + h; for h > H the
          pending events are consumed and fresh arrivals fill [H, h).
        Domain/constraints: a pending set not starting at t (relative
          tolerance 1e-9) but overlapping the request is a broken path and
          fails closed.
        """
        pending = self._jump_pending
        if pending is None:
            return self._sample_jump_events(rate_value, window_days, n_cells, dtype=dtype, device=device)
        start = float(pending["time"])
        width = float(pending["width"])
        if not self._same_time(start, time_days, window_days):
            if start < time_days + window_days and time_days < start + width:
                raise RuntimeError(
                    "Stochastic rainfall event path is inconsistent: a pending event "
                    f"set on [{start}, {start + width}] overlaps the request at "
                    f"{time_days} with width {window_days} without sharing its start."
                )
            return self._sample_jump_events(rate_value, window_days, n_cells, dtype=dtype, device=device)
        self._jump_pending = None
        events = {
            "cell_index": pending["cell_index"].to(device=device),
            "event_time": pending["event_time"].to(dtype=dtype, device=device),
            "unit_mark": pending["unit_mark"].to(dtype=dtype, device=device),
        }
        if int(pending["n_cells"]) != n_cells:
            raise RuntimeError("Stochastic rainfall pending event set does not match the cell count.")
        if self._same_time(width, window_days, window_days):
            return events
        if width > window_days:
            keep = events["event_time"] < window_days
            self._jump_pending = {
                "time": time_days + window_days,
                "width": width - window_days,
                "n_cells": n_cells,
                "cell_index": events["cell_index"][~keep].clone(),
                "event_time": (events["event_time"][~keep] - window_days).clone(),
                "unit_mark": events["unit_mark"][~keep].clone(),
            }
            return {key: value[keep] for key, value in events.items()}
        tail = self._events_on_path(
            time_days + width, window_days - width, rate_value, n_cells, dtype=dtype, device=device,
        )
        return self._concat_events(events, tail, width)

    def record_rejected_trial_path(self) -> None:
        """Pin the rejected trial's events after the substream was rewound.

        Replaying the trial from the rewound generator reproduces the same
        arrivals and marks; keeping them pending makes the shorter retry use
        exactly the events that fall inside its window, so the marked jump
        path is the same across step-size changes (the jump analogue of the
        Brownian tree).
        """
        spec = self._trial_jump_spec
        if spec is None:
            return
        events = self._events_on_path(
            spec["time"], spec["width"], spec["rate"], spec["n_cells"],
            dtype=spec["dtype"], device=spec["device"],
        )
        self._jump_pending = {
            "time": spec["time"], "width": spec["width"], "n_cells": spec["n_cells"],
            "cell_index": events["cell_index"].clone(),
            "event_time": events["event_time"].clone(),
            "unit_mark": events["unit_mark"].clone(),
        }
        self._trial_jump_spec = None

    def get_extra_state(self) -> Dict[str, Any]:
        extra_state = super().get_extra_state()
        pending = self._jump_pending
        extra_state["jump_pending"] = None if pending is None else {
            "time": float(pending["time"]),
            "width": float(pending["width"]),
            "n_cells": int(pending["n_cells"]),
            "cell_index": pending["cell_index"].detach().cpu().clone(),
            "event_time": pending["event_time"].detach().cpu().clone(),
            "unit_mark": pending["unit_mark"].detach().cpu().clone(),
        }
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping) or "jump_pending" not in state:
            raise ValueError(f"{self.SUBSTREAM_LABEL} extra state must carry jump_pending.")
        pending = state["jump_pending"]
        if pending is not None:
            expected = {"time", "width", "n_cells", "cell_index", "event_time", "unit_mark"}
            if not isinstance(pending, Mapping) or set(pending) != expected:
                raise ValueError(f"{self.SUBSTREAM_LABEL} jump_pending must carry {sorted(expected)}.")
            for key in ("cell_index", "event_time", "unit_mark"):
                if not isinstance(pending[key], Tensor):
                    raise TypeError(f"{self.SUBSTREAM_LABEL} jump_pending {key} must be a tensor.")
            width = pending["width"]
            if isinstance(width, bool) or not isinstance(width, (int, float)) or not float(width) > 0.0:
                raise ValueError(f"{self.SUBSTREAM_LABEL} jump_pending width must be positive.")
            lengths = {int(pending[key].numel()) for key in ("cell_index", "event_time", "unit_mark")}
            if len(lengths) != 1:
                raise ValueError(f"{self.SUBSTREAM_LABEL} jump_pending tensors must have one length.")
        remaining = {key: value for key, value in state.items() if key != "jump_pending"}
        super().set_extra_state(remaining)
        self._jump_pending = None if pending is None else {
            "time": float(pending["time"]), "width": float(pending["width"]),
            "n_cells": int(pending["n_cells"]),
            "cell_index": pending["cell_index"].detach().clone(),
            "event_time": pending["event_time"].detach().clone(),
            "unit_mark": pending["unit_mark"].detach().clone(),
        }

    # ------------------------------------------------------------------
    # Scientific process
    # ------------------------------------------------------------------

    @classmethod
    def discrete_moments(
        cls,
        *,
        initial_values: Mapping[str, float],
        parameters: Mapping[str, float],
        dt_days: float,
        n_steps: int,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Return the exact moments of the count and compound-Poisson depth.

        References:
          - Karlin and Taylor (1981), compound Poisson processes;
            Rodriguez-Iturbe et al. (1999).
        Mathematical expression:
          E[N_n] = N_0 + n lambda dt, Var[N_n] = n lambda dt;
          E[H_n] = H_0 + n lambda alpha dt, Var[H_n] = n lambda E[z^2] dt
          = 2 n lambda alpha^2 dt for exponential marks.
        Parameter meanings:
          lambda arrival rate, alpha mean mark, n step index 0..n_steps, dt in
          days, N_0 and H_0 the initial count and depth.
        Units:
          count mean 1 and variance 1; depth mean mm and variance mm2.
        Domain/constraints:
          lambda > 0, alpha > 0, dt > 0, n_steps >= 0.
        Sign convention:
          both means are nondecreasing in n.
        """
        rate = float(parameters["event_rate"])
        mean_depth = float(parameters["mean_depth"])
        if not (rate > 0.0 and mean_depth > 0.0 and dt_days > 0.0):
            raise ValueError("Discrete moments require positive lambda, alpha, and dt.")
        index = torch.arange(int(n_steps) + 1, dtype=torch.float64)
        expected_arrivals = rate * dt_days * index
        return {
            cls.COUNT_VARIABLE: {
                "mean": float(initial_values[cls.COUNT_VARIABLE]) + expected_arrivals,
                "variance": expected_arrivals.clone(),
            },
            cls.DEPTH_VARIABLE: {
                "mean": float(initial_values[cls.DEPTH_VARIABLE]) + mean_depth * expected_arrivals,
                "variance": 2.0 * mean_depth * mean_depth * expected_arrivals,
            },
        }

    def _arrival_process(self, event_rate_per_day: float) -> GillespieSSA:
        """Return the direct-SSA owner of the single zero-order arrival channel.

        References: Gillespie (1977); Karlin and Taylor (1981).
        Mathematical expression: a(x) = lambda (constant hazard), tau ~ Exp(lambda).
        Parameter meanings: lambda arrival rate per day on the SSA clock.
        Units: lambda day-1; SSA time in days.
        Domain/constraints: lambda finite and positive; the system is rebuilt
          only when lambda changes.
        Sign convention: the channel only adds arrivals.
        """
        if self._arrival_ssa is None or self._arrival_rate != event_rate_per_day:
            system = ReactionSystem(
                species=[self.EVENT_SPECIES],
                reactions=[
                    Reaction(
                        name=self.ARRIVAL_REACTION,
                        reactants={},
                        products={self.EVENT_SPECIES: 1},
                        rate_constant=event_rate_per_day,
                    )
                ],
                initial_state={self.EVENT_SPECIES: 0},
            )
            self._arrival_ssa = GillespieSSA(system)
            self._arrival_rate = event_rate_per_day
        return self._arrival_ssa

    @classmethod
    def arrival_cap(cls, expected_arrivals: float) -> int:
        """Return the per-cell arrival cap for one step or fail closed.

        References: Chernoff bound for Poisson tails.
        Mathematical expression: cap = ceil(mu + s sqrt(mu) + b).
        Parameter meanings: mu = lambda dt expected arrivals; s and b the
          class constants ARRIVAL_CAP_SIGMAS and ARRIVAL_CAP_BASE.
        Units: dimensionless counts.
        Domain/constraints: mu finite and nonnegative; cap <= MAX_EVENTS_PER_STEP.
        Sign convention: not applicable.
        """
        if not math.isfinite(expected_arrivals) or expected_arrivals < 0.0:
            raise ValueError("Expected arrivals must be finite and nonnegative.")
        cap = int(
            math.ceil(
                expected_arrivals
                + cls.ARRIVAL_CAP_SIGMAS * math.sqrt(expected_arrivals)
                + cls.ARRIVAL_CAP_BASE
            )
        )
        if cap > cls.MAX_EVENTS_PER_STEP:
            raise ValueError(
                f"Expected arrivals {expected_arrivals:.6g} per step exceed the admissible "
                f"SSA arrival cap {cls.MAX_EVENTS_PER_STEP}; reduce dt."
            )
        return cap

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """Emit the arrivals and depth marks of one fixed step.

        References:
          - Rodriguez-Iturbe et al. (1999); Laio et al. (2001); Gillespie (1977).
        Mathematical expression:
          k ~ Poisson(lambda dt_d) sampled as direct-SSA arrivals in [0, dt_d);
          Delta N = k; Delta H = alpha sum_{i<=k} E_i with E_i ~ Exp(1);
          compensators lambda dt_d and lambda alpha dt_d.
        Parameter meanings:
          lambda arrival rate, alpha mean depth, E_i unit exponential draws,
          dt_d = dt / SECONDS_PER_DAY.
        Units:
          N and Delta N 1; H and Delta H mm; lambda day-1; alpha mm; dt s.
        Domain/constraints:
          both states finite tensors of equal shape, count integer-valued and
          nonnegative; finite positive dt; every emitted step consumes the SSA
          and mark draws of exactly this step; more than MAX_EVENTS_PER_STEP
          arrivals in one cell is rejected.
        Sign convention:
          increments are nonnegative; the depth gradient with respect to alpha
          is the sum of unit marks (Delta H / alpha).
        """
        # References: Rodriguez-Iturbe et al. (1999); Laio et al. (2001); Gillespie
        #   (1977); Karlin and Taylor (1981) compound Poisson processes.
        # Mathematical expression: Delta N_c = #{arrivals in [t, t + dt_d)} and
        #   Delta H_c = alpha sum_i E_i over those arrivals, E_i ~ Exp(1); the event
        #   set comes from the Functor-owned event path ledger.
        # Parameter meanings: lambda arrival rate, alpha mean depth, E_i unit marks,
        #   dt_d step in days.
        # Units: N 1; H mm; lambda day-1; alpha mm; dt_d day; compensators 1 and mm.
        # Domain/constraints: lambda, alpha > 0; finite positive dt; arrival cap is a
        #   fail-closed guard; events keep their marks across step-size changes.
        # Sign convention: arrivals only add counts and nonnegative depth.

        if kwargs:
            raise TypeError(f"Unsupported stochastic rainfall forward arguments: {sorted(kwargs)}")
        for name in self.STATE_VARIABLES:
            if name not in state:
                raise ValueError(f"{name} is required for the stochastic rainfall process")
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"State '{name}' must be a torch.Tensor")
            if not value.is_floating_point():
                raise ValueError(f"State '{name}' must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"State '{name}' must contain only finite values")
        count = state[self.COUNT_VARIABLE]
        depth = state[self.DEPTH_VARIABLE]
        if count.shape != depth.shape:
            raise ValueError("Rainfall count and depth states must share one shape")
        if not torch.equal(count, torch.round(count)) or bool((count < 0).any()):
            raise ValueError(f"State '{self.COUNT_VARIABLE}' must be an integer-valued nonnegative count")
        dt_value = float(dt)
        if not torch.isfinite(count.new_tensor(dt_value)) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")
        dt_days = dt_value / float(SECONDS_PER_DAY)

        event_rate = self.get_parameter_value("event_rate", required=True).to(count)
        mean_depth = self.get_parameter_value("mean_depth", required=True).to(count)
        if event_rate.numel() != 1 or mean_depth.numel() != 1:
            raise ValueError("Stochastic rainfall parameters must be scalar.")
        rate_value = float(event_rate.detach())
        if not rate_value > 0.0 or not float(mean_depth.detach()) > 0.0:
            raise ValueError("Stochastic rainfall event_rate and mean_depth must be positive.")
        self._require_substream_device(count)

        # References: Gillespie (1977) direct method with one constant channel.
        # Mathematical expression: arrivals in [0, dt_d) with Exp(lambda) gaps.
        # Parameter meanings: lambda per day; final_time dt_d; one sample per cell.
        # Units: SSA clock in days.
        # Domain/constraints: a cell reaching MAX_EVENTS_PER_STEP fails closed.
        # Sign convention: trajectory_lengths - 1 arrivals per cell, never negative.
        n_cells = count.numel()
        time_days = float(time) / float(SECONDS_PER_DAY)
        self._trial_jump_spec = {
            "time": time_days, "width": dt_days, "rate": rate_value, "n_cells": n_cells,
            "dtype": count.dtype, "device": count.device,
        }
        events = self._events_on_path(
            time_days, dt_days, rate_value, n_cells, dtype=count.dtype, device=count.device,
        )
        arrivals = torch.bincount(events["cell_index"], minlength=n_cells)

        # References: Rodriguez-Iturbe et al. (1999) exponential depth marks.
        # Mathematical expression: Delta H_c = alpha sum_{i in cell c} E_i, E_i ~ Exp(1).
        # Parameter meanings: alpha keeps its autograd graph; E_i are unit draws.
        # Units: E_i 1; Delta H mm.
        # Domain/constraints: marks stay attached to their events across
        # step-size changes through the event path ledger.
        # Sign convention: nonnegative marks.
        unit_sum = torch.zeros(n_cells, dtype=count.dtype, device=count.device)
        if int(events["cell_index"].numel()) > 0:
            unit_sum.index_add_(0, events["cell_index"], events["unit_mark"])
        depth_increment = mean_depth.reshape(()) * unit_sum
        self.draw_count += 1

        count_increment = arrivals.to(dtype=count.dtype).reshape(count.shape)
        depth_increment = depth_increment.reshape(count.shape)
        compensator_count = (event_rate.reshape(()) * dt_days).detach().expand(count.shape).clone()
        compensator_depth = (
            (event_rate.reshape(()) * mean_depth.reshape(()) * dt_days).detach().expand(count.shape).clone()
        )

        output: FluxDict = {}
        for key, value, unit in (
            (self.COUNT_VARIABLE, count_increment, "1"),
            (self.DEPTH_VARIABLE, depth_increment, "mm"),
        ):
            output[key] = UpdateItem(
                value=value,
                mode=UpdateMode.ACCUMULATE,
                unit=unit,
                per_timestep=True,
                source_functor=self.name,
            )
        for key, value, unit in (
            (self.DIAGNOSTICS[0], count_increment.detach().clone(), "1"),
            (self.DIAGNOSTICS[1], depth_increment.detach().clone(), "mm"),
            (self.DIAGNOSTICS[2], compensator_count, "1"),
            (self.DIAGNOSTICS[3], compensator_depth, "mm"),
        ):
            output[key] = UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=True,
                source_functor=self.name,
            )
        return output


__all__ = ["MarkedPoissonRainfallFunctor"]
