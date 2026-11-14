"""Ornstein-Uhlenbeck stochastic air-temperature anomaly (continuous SDE).

Theory document: docs/source/theory/atmosphere_stochastic_temperature.md

The Functor advances one prognostic anomaly state with the existing
fixed-step Euler-Maruyama owner (relic.stochastic.sde_solver) and a
Functor-owned Wiener substream whose position is part of the module
extra-state, so accepted-step rollback and checkpoint restart reproduce the
same path.  The noise is an external forcing closure (Hasselmann 1976); no
fluctuation-dissipation relation is claimed.
"""

from __future__ import annotations

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
from ...stochastic.sde_solver import EulerMaruyamaSolver, SDEParameters
from ...utils.time_constants import SECONDS_PER_DAY

class OrnsteinUhlenbeckTemperatureFunctor(FunctorSubstreamMixin, NamespaceDefaultMixin, PhysicalFieldFunctor):
    """Continuous-SDE anomaly dX = -alpha X dt + sigma dW (PhysicalField contract).

    References:
      - Hasselmann (1976), Tellus 28, 473-485.
      - Gardiner (2009), Stochastic Methods, ch. 4.
      - Kloeden and Platen (1992), ch. 10; Higham (2001), SIAM Review 43.
    Mathematical expression:
      X_{n+1} = X_n - alpha X_n dt + sigma dW_n, dW_n ~ N(0, dt).
    Parameter meanings:
      X anomaly state, alpha mean reversion rate, sigma volatility, dt the
      simulator step converted to days, W the Functor-owned Wiener substream.
    Units:
      X K; alpha day-1; sigma K day-0.5; dt s -> day; drift K day-1;
      wiener_increment day-0.5 (sqrt(dt) times a standard normal draw).
    Domain/constraints:
      alpha > 0, sigma > 0 (ParameterManager bounds); 0 < alpha dt < 1 for the
      explicit step; finite floating-point state; diagonal diffusion only.
    Sign convention:
      a positive increment warms the anomaly; the drift relaxes toward zero.
    """

    parameter_namespace = "stochastic.weather.temperature"
    STATE_VARIABLE = "stochastic_air_temperature_anomaly"
    DIAGNOSTICS = (
        "diagnostics.stochastic_temperature.drift",
        "diagnostics.stochastic_temperature.diffusion",
        "diagnostics.stochastic_temperature.wiener_increment",
    )
    EXTRA_STATE_FORMAT = "relic.stochastic_temperature.wiener_substream"
    EXTRA_STATE_VERSION = 1
    SUBSTREAM_LABEL = "Stochastic temperature"
    stochastic_category = "continuous_sde"
    noise_classification = "external_forcing"
    solver_name = "euler_maruyama"
    # Override whitelist and runtime short names (namespace parameter_namespace);
    # metadata.parameters declares the full ParameterManager keys.
    PARAMETERS = {
        "reversion_rate": "Mean reversion rate alpha [day-1]",
        "volatility": "Volatility sigma [K day-0.5]",
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
        name: str = "stochastic_air_temperature",
        *,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Stochastic temperature config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        seed = self.validate_substream_seed(config_dict.pop("seed", None))
        parameter_overrides = config_dict.pop("parameters", None)
        if parameter_overrides is None:
            parameter_overrides = {}
        if not isinstance(parameter_overrides, Mapping):
            raise ValueError("Stochastic temperature parameter overrides must be a mapping.")
        device = kwargs.pop("device", None)

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Ornstein-Uhlenbeck air-temperature anomaly advanced by fixed-step "
                "Euler-Maruyama with a Functor-owned Wiener substream"
            ),
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "dX = -alpha X dt + sigma dW; "
                "X_{n+1} = X_n - alpha X_n dt + sigma dW_n, dW_n ~ N(0, dt)"
            ),
            references=[
                "Hasselmann, K. (1976). Tellus 28, 473-485",
                "Gardiner, C. (2009). Stochastic Methods, 4th ed., Springer",
                "Kloeden, P.E. & Platen, E. (1992). Numerical Solution of SDEs, Springer",
                "Higham, D.J. (2001). SIAM Review 43, 525-546",
            ],
            parameters={
                f"{self.parameter_namespace}.{short_name}": description
                for short_name, description in self.PARAMETERS.items()
            },
            input_variables=[self.STATE_VARIABLE],
            output_variables=[self.STATE_VARIABLE, *self.DIAGNOSTICS],
            conservation_laws=["none"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            coupling_interfaces=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        unknown = sorted(set(parameter_overrides).difference(self.PARAMETERS))
        if unknown:
            raise ValueError("Unknown stochastic temperature parameter override(s): " + ", ".join(unknown))
        # Explicit overrides keep the caller's tensor (and its autograd graph)
        # through the canonical {"value": ...} config form.
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
        # The Wiener substream (generator, seed, draw_count, extra state, and
        # device migration) is owned through FunctorSubstreamMixin.
        self._initialize_substream(seed)
        for parameter_name in self.PARAMETERS:
            self.get_parameter_value(parameter_name, required=True)
        if device is not None:
            self.to_device(device)

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
        """Return the exact conditional moments of the emitted AR(1) recursion.

        References:
          - Hasselmann (1976); Gardiner (2009) ch. 4; Kloeden and Platen (1992).
        Mathematical expression:
          phi = 1 - alpha dt; E[X_n | X_0] = phi^n X_0;
          Var[X_n | X_0] = sigma^2 dt (1 - phi^{2n}) / (1 - phi^2).
        Parameter meanings:
          alpha reversion rate, sigma volatility, X_0 initial anomaly, n step
          index 0..n_steps, dt in days.
        Units:
          mean K; variance K2; dt day; alpha day-1; sigma K day-0.5.
        Domain/constraints:
          0 < alpha dt < 1 (the explicit step admissibility enforced by
          forward); n_steps >= 0.
        Sign convention:
          the mean relaxes toward zero with the sign of X_0.
        """
        alpha = float(parameters["reversion_rate"])
        sigma = float(parameters["volatility"])
        phi = 1.0 - alpha * dt_days
        if not 0.0 < phi < 1.0:
            raise ValueError("Discrete moments require 0 < alpha dt < 1.")
        x0 = float(initial_values[cls.STATE_VARIABLE])
        index = torch.arange(int(n_steps) + 1, dtype=torch.float64)
        mean = x0 * phi**index
        variance = sigma * sigma * dt_days * (1.0 - phi ** (2.0 * index)) / (1.0 - phi * phi)
        return {cls.STATE_VARIABLE: {"mean": mean, "variance": variance}}

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """Advance the anomaly by one fixed Euler-Maruyama step.

        References:
          - Hasselmann (1976); Kloeden and Platen (1992) ch. 10; Higham (2001).
        Mathematical expression:
          Delta X = -alpha X dt_d + sigma dW, dt_d = dt / SECONDS_PER_DAY,
          dW = Delta W([t_d, t_d + dt_d]) from the Functor-owned Brownian
          tree: sqrt(dt_d) xi for a fresh interval, or the bridge-conditioned
          refinement of an increment already fixed on the path by a rejected
          adaptive trial (Gaines and Lyons 1997).
        Parameter meanings:
          alpha reversion rate, sigma volatility, xi standard-normal draw,
          t_d the step start in days.
        Units:
          X and Delta X K; dt s; dt_d day; alpha day-1; sigma K day-0.5;
          drift K day-1; diffusion K day-0.5; wiener_increment day-0.5.
        Domain/constraints:
          finite floating-point state tensor; finite positive dt;
          0 < alpha dt_d < 1 (explicit step); one draw per emitted step.
        Sign convention:
          positive increment warms the anomaly; drift relaxes toward zero.
        """
        # References: Hasselmann (1976); Kloeden and Platen (1992) ch. 10; Higham
        #   (2001); Gaines and Lyons (1997) Brownian tree.
        # Mathematical expression: Delta X = -alpha X dt_d + sigma Delta W with
        #   Delta W = Delta W([t_d, t_d + dt_d]) from the Functor-owned path.
        # Parameter meanings: alpha reversion rate, sigma volatility, dt_d step in
        #   days, Delta W the path-consistent Wiener increment.
        # Units: X and Delta X K; alpha day-1; sigma K day-0.5; dt_d day; Delta W day-0.5.
        # Domain/constraints: finite floating state; 0 < alpha dt_d < 1; one emitted
        #   step per call; the increment survives adaptive step-size changes.
        # Sign convention: positive increment warms the anomaly; drift relaxes to zero.

        if kwargs:
            raise TypeError(f"Unsupported stochastic temperature forward arguments: {sorted(kwargs)}")
        if self.STATE_VARIABLE not in state:
            raise ValueError(f"{self.STATE_VARIABLE} is required for the stochastic temperature anomaly")
        anomaly = state[self.STATE_VARIABLE]
        if not isinstance(anomaly, Tensor):
            raise TypeError(f"State '{self.STATE_VARIABLE}' must be a torch.Tensor")
        if not anomaly.is_floating_point():
            raise ValueError(f"State '{self.STATE_VARIABLE}' must be a floating-point tensor")
        if not bool(torch.isfinite(anomaly).all()):
            raise ValueError(f"State '{self.STATE_VARIABLE}' must contain only finite values")
        dt_value = float(dt)
        if not torch.isfinite(anomaly.new_tensor(dt_value)) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")
        dt_days = dt_value / float(SECONDS_PER_DAY)

        alpha = self.get_parameter_value("reversion_rate", required=True).to(anomaly)
        sigma = self.get_parameter_value("volatility", required=True).to(anomaly)
        if alpha.numel() != 1 or sigma.numel() != 1:
            raise ValueError("Stochastic temperature parameters must be scalar.")
        alpha_dt = float(alpha.detach()) * dt_days
        if not 0.0 < alpha_dt < 1.0:
            raise ValueError(
                "Explicit Euler-Maruyama step requires 0 < alpha dt < 1; "
                f"got alpha dt = {alpha_dt:.6g} for dt = {dt_value:.6g} s."
            )
        self._require_substream_device(anomaly)
        time_days = float(time) / float(SECONDS_PER_DAY)
        wiener_increment = self.draw_wiener_increment(time_days, dt_days, like=anomaly)

        def drift(x: Tensor, t: Tensor) -> Tensor:
            return -alpha.reshape(()) * x

        def diffusion(x: Tensor, t: Tensor) -> Tensor:
            return sigma.reshape(()).expand_as(x)

        solver = EulerMaruyamaSolver(
            SDEParameters(dt=dt_days, method="euler_maruyama", diffusion_structure="diagonal"),
            drift_func=drift,
            diffusion_func=diffusion,
        )
        sample = solver.sample(
            anomaly,
            num_samples=1,
            num_steps=1,
            generator=self.generator,
            initial_time=anomaly.new_tensor(time_days),
            wiener_increments=wiener_increment.detach().reshape(1, 1, *anomaly.shape),
        )
        self.draw_count += 1
        solution = sample["solution"]
        increment = solution[0, 1] - solution[0, 0]
        drift_value = sample["drift_history"][0, 0]
        diffusion_value = sample["diffusion_history"][0, 0]
        wiener_increment = wiener_increment.detach()

        output: FluxDict = {
            self.STATE_VARIABLE: UpdateItem(
                value=increment,
                mode=UpdateMode.ACCUMULATE,
                unit="K",
                per_timestep=True,
                source_functor=self.name,
            ),
        }
        for key, value, unit in (
            (self.DIAGNOSTICS[0], drift_value, "K day-1"),
            (self.DIAGNOSTICS[1], diffusion_value, "K day-0.5"),
            (self.DIAGNOSTICS[2], wiener_increment, "day-0.5"),
        ):
            output[key] = UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=True,
                source_functor=self.name,
            )
        return output


__all__ = ["OrnsteinUhlenbeckTemperatureFunctor"]
