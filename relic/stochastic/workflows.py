"""Workflow helpers for stochastic CLI and API execution."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Any, List, Mapping, Optional, Iterable, Tuple

import torch
import xarray as xr
from torch import Tensor

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.parameters import ParameterSpec, get_global_manager
from relic.scientific.atmosphere.stochastic_rainfall import MarkedPoissonRainfallFunctor
from relic.scientific.atmosphere.stochastic_temperature import (
    OrnsteinUhlenbeckTemperatureFunctor,
)
from relic.stochastic.config import load_config, get_section
from relic.stochastic.gillespie import GillespieSSA
from relic.stochastic.io import save_results
from relic.stochastic.sde_solver import EulerMaruyamaSolver
from relic.utils.time_constants import SECONDS_PER_DAY
from relic.stochastic.distributions import (
    ParameterDistribution,
    DistributionSpec,
    DistributionType,
    CorrelationKernel,
)
from relic.stochastic.validation import pit_test, crps_score, coverage_probability

logger = logging.getLogger(__name__)

_ENSEMBLE_DIMENSION_NAMES = frozenset(
    {
        "ensemble",
        "ensemble_member",
        "ensemble_members",
        "member",
        "members",
        "realization",
        "realizations",
        "sample",
        "samples",
        "trajectory",
        "trajectories",
    }
)


def _prepare_validation_ensemble(
    observations: torch.Tensor,
    predictions: torch.Tensor,
    prediction_dims: Optional[Tuple[str, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize validation data to [observation, ensemble_member]."""
    observations = torch.as_tensor(observations)
    predictions = torch.as_tensor(predictions)
    if observations.dim() != 1:
        raise ValueError(
            f"Validation observations must be 1D, got {observations.shape}"
        )
    if predictions.dim() != 2:
        raise ValueError(
            "Validation predictions must be 2D with observation and "
            f"ensemble axes, got {predictions.shape}"
        )
    observation_count = observations.shape[0]
    member_axis = None
    if prediction_dims is not None:
        if len(prediction_dims) != predictions.dim():
            raise ValueError(
                "Validation prediction dimensions do not match tensor rank: "
                f"{prediction_dims} vs {predictions.shape}"
            )
        member_axes = [
            axis
            for axis, name in enumerate(prediction_dims)
            if name.lower() in _ENSEMBLE_DIMENSION_NAMES
        ]
        if len(member_axes) > 1:
            raise ValueError(
                "Validation prediction dimensions identify multiple "
                f"ensemble axes: {prediction_dims}"
            )
        if member_axes:
            member_axis = member_axes[0]

    if member_axis is not None:
        observation_axis = 1 - member_axis
        if predictions.shape[observation_axis] != observation_count:
            raise ValueError(
                "Validation prediction observation axis does not match "
                f"observation count: {predictions.shape} vs {observation_count}"
            )
        ensemble = (
            predictions
            if observation_axis == 0
            else predictions.transpose(0, 1)
        )
    elif (
        predictions.shape[0] == observation_count
        and predictions.shape[1] == observation_count
    ):
        raise ValueError(
            "Validation prediction axes are ambiguous for a square tensor; "
            "use NetCDF ensemble dimension metadata"
        )
    elif predictions.shape[0] == observation_count:
        ensemble = predictions
    elif predictions.shape[1] == observation_count:
        ensemble = predictions.transpose(0, 1)
    else:
        raise ValueError(
            "Validation prediction shape does not match observation count: "
            f"{predictions.shape} vs {observation_count}"
        )
    if not observations.is_floating_point():
        observations = observations.float()
    if not ensemble.is_floating_point():
        ensemble = ensemble.float()
    return observations, ensemble


def _summarize_validation_results(
    pit_result: Any,
    crps_result: Any,
    coverage_result: Any,
    nominal_coverage: float,
) -> Dict[str, float]:
    """Convert validation result objects to serializable scalar metrics."""
    return {
        "pit_mean": float(pit_result.score),
        "pit_pvalue": (
            float(pit_result.pvalue)
            if pit_result.pvalue is not None
            else float("nan")
        ),
        "crps_mean": float(crps_result.score),
        "coverage_probability": float(coverage_result.score),
        "coverage_pvalue": (
            float(coverage_result.pvalue)
            if coverage_result.pvalue is not None
            else float("nan")
        ),
        "nominal_coverage": float(nominal_coverage),
        "sample_size": int(pit_result.sample_size),
    }


def _as_tuple_bounds(value: Optional[Iterable[float]]) -> Tuple[float, float]:
    if value is None:
        return (-float("inf"), float("inf"))
    bounds = list(value)
    if len(bounds) != 2:
        raise ValueError("Bounds must have exactly two values")
    return (float(bounds[0]), float(bounds[1]))


def _pick_numeric(config: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return float(value)
    return None


def _build_normal_like_params(
    param_name: str,
    param_config: Dict[str, Any],
    *,
    error_message: str,
) -> Dict[str, float]:
    mean = _pick_numeric(param_config, "mean", "loc")
    std = _pick_numeric(param_config, "std", "scale")
    if mean is None or std is None:
        raise ValueError(error_message.format(param_name=param_name))
    return {"loc": float(mean), "scale": float(std)}


def _build_uniform_params(param_name: str, param_config: Dict[str, Any]) -> Dict[str, float]:
    low = _pick_numeric(param_config, "low", "min")
    high = _pick_numeric(param_config, "high", "max")
    if low is None or high is None:
        raise ValueError(f"Uniform distribution for {param_name} requires low/high")
    return {"low": float(low), "high": float(high)}


def _build_gamma_params(param_name: str, param_config: Dict[str, Any]) -> Dict[str, float]:
    concentration = _pick_numeric(param_config, "concentration", "shape")
    rate = _pick_numeric(param_config, "rate")
    if concentration is None or rate is None:
        raise ValueError(f"Gamma distribution for {param_name} requires concentration/rate")
    return {"concentration": float(concentration), "rate": float(rate)}


def _build_beta_params(param_name: str, param_config: Dict[str, Any]) -> Dict[str, float]:
    alpha = _pick_numeric(param_config, "alpha")
    beta = _pick_numeric(param_config, "beta")
    if alpha is None or beta is None:
        raise ValueError(f"Beta distribution for {param_name} requires alpha/beta")
    params: Dict[str, float] = {"alpha": float(alpha), "beta": float(beta)}
    if "low" in param_config:
        params["low"] = float(param_config["low"])
    if "high" in param_config:
        params["high"] = float(param_config["high"])
    return params


def _build_triangular_params(param_name: str, param_config: Dict[str, Any]) -> Dict[str, float]:
    params = _build_uniform_params(param_name, param_config)
    if "mode" in param_config:
        params["mode"] = float(param_config["mode"])
    return params


def _build_params_from_legacy_fields(
    dist_type: DistributionType,
    param_name: str,
    param_config: Dict[str, Any],
) -> Dict[str, float]:
    if dist_type == DistributionType.NORMAL:
        return _build_normal_like_params(
            param_name,
            param_config,
            error_message="Normal distribution for {param_name} requires mean/std",
        )
    if dist_type == DistributionType.UNIFORM:
        return _build_uniform_params(param_name, param_config)
    if dist_type == DistributionType.LOGNORMAL:
        return _build_normal_like_params(
            param_name,
            param_config,
            error_message="Lognormal distribution for {param_name} requires mean/std",
        )
    if dist_type == DistributionType.GAMMA:
        return _build_gamma_params(param_name, param_config)
    if dist_type == DistributionType.BETA:
        return _build_beta_params(param_name, param_config)
    if dist_type == DistributionType.TRIANGULAR:
        return _build_triangular_params(param_name, param_config)
    if dist_type == DistributionType.TRUNCATED_NORMAL:
        return _build_normal_like_params(
            param_name,
            param_config,
            error_message="Truncated normal for {param_name} requires mean/std",
        )
    raise ValueError(
        f"No parameter fields are defined for distribution type "
        f"'{dist_type.value}' of {param_name}"
    )


def _parse_distribution_spec(param_name: str, param_config: Dict[str, Any]) -> DistributionSpec:
    dist_name = param_config.get("distribution") or param_config.get("dist_type")
    if not dist_name:
        raise ValueError(f"Parameter {param_name} missing distribution type")

    try:
        dist_type = DistributionType(str(dist_name).lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported distribution type '{dist_name}' for {param_name}") from exc

    params = param_config.get("params") or {}
    if not params:
        params = _build_params_from_legacy_fields(dist_type, param_name, param_config)

    bounds = _as_tuple_bounds(param_config.get("bounds"))
    log_scale = bool(param_config.get("log_scale", False))
    integer_valued = bool(param_config.get("integer_valued", False))

    # Kernel and nugget are passed exactly as the configuration states them;
    # DistributionSpec requires both for a correlated field and neither for
    # an uncorrelated one (no kernel or nugget is chosen here).
    kernel = None
    if "kernel" in param_config:
        kernel_value = param_config["kernel"]
        try:
            kernel = CorrelationKernel(str(kernel_value).lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported kernel '{kernel_value}' for {param_name}") from exc
    nugget = None
    if "correlation_nugget" in param_config:
        nugget = float(param_config["correlation_nugget"])

    return DistributionSpec(
        dist_type=dist_type,
        params=params,
        bounds=bounds,
        log_scale=log_scale,
        integer_valued=integer_valued,
        correlation_length=param_config.get("correlation_length"),
        kernel=kernel,
        correlation_nugget=nugget,
    )


# Keys an uncertainty parameter entry may carry (DistributionSpec fields,
# the legacy per-family fields read by _build_params_from_legacy_fields,
# and the ParameterManager selectors). Any other key is rejected so that a
# misspelled option is never ignored.
_UNCERTAINTY_PARAMETER_KEYS = frozenset({
    "distribution", "dist_type", "params", "bounds", "log_scale",
    "integer_valued", "kernel", "correlation_length", "correlation_nugget",
    "mean", "loc", "std", "scale", "low", "min", "high", "max",
    "concentration", "shape", "rate", "alpha", "beta", "mode",
    "use_parameter_manager", "from_parameter_manager", "key", "units",
})


def _declared_sample_units(
    parameters_section: Dict[str, Any], specs: Dict[str, Any]
) -> Dict[str, str]:
    """Units of each sampled parameter: the ParameterSpec units or the declared 'units'.

    A parameter without either has no entry; writing its samples to NetCDF
    then raises in relic.stochastic.io instead of guessing a unit.
    """
    units: Dict[str, str] = {}
    for param_name, spec in specs.items():
        if isinstance(spec, ParameterSpec):
            units[param_name] = str(spec.units)
            continue
        declared = (parameters_section.get(param_name) or {}).get("units")
        if declared is not None:
            units[param_name] = str(declared)
    return units


def _build_parameter_specs(parameters_section: Dict[str, Any]) -> Dict[str, Any]:
    specs: Dict[str, Any] = {}
    param_manager = get_global_manager()

    for param_name, param_config in parameters_section.items():
        if param_config is None:
            param_config = {}
        if not isinstance(param_config, dict):
            raise ValueError(f"Parameter config for {param_name} must be a dict")
        unknown_keys = sorted(set(param_config) - _UNCERTAINTY_PARAMETER_KEYS)
        if unknown_keys:
            raise ValueError(
                f"Unknown uncertainty setting(s) {unknown_keys} for {param_name}"
            )

        use_manager = bool(param_config.get("use_parameter_manager") or param_config.get("from_parameter_manager"))
        dist_name = param_config.get("distribution") or param_config.get("dist_type")

        if use_manager or (not dist_name and "params" not in param_config):
            spec_key = param_config.get("key", param_name)
            spec = param_manager.get_parameter(spec_key)
            if spec is None:
                raise ValueError(f"ParameterSpec not found for {spec_key}")
            specs[param_name] = spec
            continue

        specs[param_name] = _parse_distribution_spec(param_name, param_config)

    return specs


def run_uncertainty_analysis(args) -> None:
    """Run parameter uncertainty sampling."""
    try:
        config = load_config(args.config) if args.config else {}
        section = get_section(config, "uncertainty")
        # A CLI value overrides the config value; an explicit 0 is not read as
        # "absent" (it is rejected by the sampler instead).
        n_samples = args.samples if args.samples is not None else section.get("samples")
        if n_samples is None:
            raise ValueError("uncertainty.samples is required")

        parameters_section = section.get("parameters", {})
        if not isinstance(parameters_section, dict) or not parameters_section:
            raise ValueError("uncertainty.parameters must be a non-empty mapping")

        # References:
        #   Saltelli et al. (2008); Rasmussen and Williams (2006).
        # Formula:
        #   theta_i ~ p_i(theta) for i=1..N, with optional spatial correlation K(x,x').
        # Parameters:
        #   distribution type, params (loc/scale or low/high), bounds, correlation_length, kernel.
        parameter_specs = _build_parameter_specs(parameters_section)

        spatial_coords = None
        if "spatial_coords" in section:
            spatial_coords = torch.tensor(section["spatial_coords"], dtype=torch.float32)

        distribution = ParameterDistribution(
            parameter_specs=parameter_specs,
            spatial_coords=spatial_coords,
        )
        samples = distribution.sample(
            input_state=torch.tensor(0.0),
            num_samples=int(n_samples),
            seed=(
                args.seed
                if getattr(args, "seed", None) is not None
                else section.get("seed")
            ),
        )

        sample_units = _declared_sample_units(parameters_section, parameter_specs)
        payload: Dict[str, Any] = dict(samples)
        payload["variable_units"] = sample_units
        payload["variable_dimensions"] = {name: ["sample", "location"] for name in sample_units}

        output_path = args.output or "uncertainty_results"
        save_results(payload, output_path, args.format)
        logger.info("Uncertainty sampling completed: %s samples", n_samples)

    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
        logger.error("Uncertainty analysis failed: %s", exc)
        raise


def run_weather_generation(args) -> None:
    """Reject weather execution until the canonical forcing binding exists.

    References:
        Hasselmann (1976); Wilks (2006); revised foundations, stochastic
        forcing and claim-owner-evidence contracts.
    Mathematical expression:
        dT = a(T,t) dt + b(T,t) dW and marked precipitation events are
        executable only after their operators and stochastic path are bound.
    Parameter meanings:
        a and b are declared temperature operators; event marks and
        intensities define precipitation; all require canonical parameter,
        unit, calendar, forcing-cursor, state, checkpoint, and RNG bindings.
    """

    del args
    raise ValueError(
        "CLI weather generation requires a declared weather-forcing operator "
        "and a validated GraphIR/Functor runtime binding. This integration is "
        "currently [N]; no standalone scientific forcing law is inferred."
    )


@dataclass(frozen=True)
class StochasticProcessBinding:
    """Executable GraphIR binding of one stochastic scientific Functor.

    The binding names the existing Functor owner, its prognostic states, the
    existing relic.stochastic algorithm owner it delegates to, and the
    executed validation test.  Coefficients, units, marks, and the random
    substream are owned by the Functor itself; this record only selects that
    owner for the public API/CLI entry points.
    """

    process: str
    semantic: str
    functor_class: type
    state_variables: tuple[str, ...]
    solver_class: type
    rng_substream: str
    validation_test: str
    term_name: str
    noise_type: str
    parameter_roles: Mapping[str, str] = field(default_factory=dict)

    @property
    def functor_class_path(self) -> str:
        return f"{self.functor_class.__module__}.{self.functor_class.__qualname__}"

    @property
    def node_type(self) -> str:
        return f"Functor:{self.functor_class_path}"

    @property
    def solver_owner(self) -> str:
        return f"{self.solver_class.__module__}.{self.solver_class.__qualname__}"

    def parameter_key(self, role_or_name: str) -> str:
        name = self.parameter_roles.get(role_or_name, role_or_name)
        return f"{self.functor_class.parameter_namespace}.{name}"


_SEMANTIC_LABELS = {
    "continuous_sde": "continuous-SDE",
    "marked_jump": "marked-jump",
}

BOUND_STOCHASTIC_PROCESSES: Dict[str, StochasticProcessBinding] = {
    "stochastic_air_temperature": StochasticProcessBinding(
        process="stochastic_air_temperature",
        semantic="continuous_sde",
        functor_class=OrnsteinUhlenbeckTemperatureFunctor,
        state_variables=(OrnsteinUhlenbeckTemperatureFunctor.STATE_VARIABLE,),
        solver_class=EulerMaruyamaSolver,
        rng_substream="stochastic_air_temperature",
        validation_test="test/unit/atmosphere/test_stochastic_temperature.py",
        term_name="air_temperature_anomaly_noise",
        noise_type="white",
    ),
    "stochastic_rainfall": StochasticProcessBinding(
        process="stochastic_rainfall",
        semantic="marked_jump",
        functor_class=MarkedPoissonRainfallFunctor,
        state_variables=MarkedPoissonRainfallFunctor.STATE_VARIABLES,
        solver_class=GillespieSSA,
        rng_substream="stochastic_rainfall",
        validation_test="test/unit/atmosphere/test_stochastic_rainfall.py",
        term_name="rainfall_arrival",
        noise_type="marked_poisson",
        parameter_roles={"propensity": "event_rate", "mark": "mean_depth"},
    ),
}

BOUND_CONTINUOUS_SDE_PROCESSES: Dict[str, StochasticProcessBinding] = {
    name: binding
    for name, binding in BOUND_STOCHASTIC_PROCESSES.items()
    if binding.semantic == "continuous_sde"
}
BOUND_MARKED_JUMP_PROCESSES: Dict[str, StochasticProcessBinding] = {
    name: binding
    for name, binding in BOUND_STOCHASTIC_PROCESSES.items()
    if binding.semantic == "marked_jump"
}

BOUND_CONTINUOUS_SDE_GRAPH_OWNER = (
    "relic.stochastic.workflows.build_bound_continuous_sde_graph"
)
BOUND_MARKED_JUMP_GRAPH_OWNER = (
    "relic.stochastic.workflows.build_bound_marked_jump_graph"
)


@dataclass(frozen=True)
class BoundStochasticGraph:
    """Registry, GraphIR, template Functor, units, and binding of one bound process."""

    binding: StochasticProcessBinding
    registry: NodeRegistry
    graph: GraphIR
    template: Any
    state_units: Dict[str, str]


def require_bound_stochastic_process(
    process: Any,
    *,
    semantic: Optional[str] = None,
) -> StochasticProcessBinding:
    """Return the binding of a bound stochastic process or fail closed."""
    name = process.strip() if isinstance(process, str) else ""
    binding = BOUND_STOCHASTIC_PROCESSES.get(name)
    if binding is None or (semantic is not None and binding.semantic != semantic):
        label = _SEMANTIC_LABELS.get(semantic or "", "stochastic")
        candidates = sorted(
            candidate
            for candidate, item in BOUND_STOCHASTIC_PROCESSES.items()
            if semantic is None or item.semantic == semantic
        )
        raise ValueError(
            f"{process!r} is not a bound {label} process; bound processes: "
            + ", ".join(candidates)
            + ". No standalone scientific formula is inferred."
        )
    return binding


def require_bound_continuous_sde_process(process: Any) -> StochasticProcessBinding:
    return require_bound_stochastic_process(process, semantic="continuous_sde")


def require_bound_marked_jump_process(process: Any) -> StochasticProcessBinding:
    return require_bound_stochastic_process(process, semantic="marked_jump")


def _reproducibility_map(binding: StochasticProcessBinding, template: Any, *, seed: int) -> Dict[str, Any]:
    return {
        "random_seed": int(seed),
        "noise_type": binding.noise_type,
        "solver_method": template.solver_name,
        "uses_ensemble": False,
        "checkpoint_owner": "relic.core.simulator.checkpoint",
        "rng_state_owner": "relic.core.simulator.checkpoint",
        "rng_substream": binding.rng_substream,
        "restart_policy": "restore_and_replay",
        "time_step_mode": "fixed",
    }


def _execution_binding(binding: StochasticProcessBinding) -> Dict[str, Any]:
    return {
        "functor_class": binding.functor_class_path,
        "forward_owner": binding.functor_class_path + ".forward",
        "solver_owner": binding.solver_owner,
        "state_vars": list(binding.state_variables),
        "rng_state_scope": "functor_extra_state",
        "validation_test": binding.validation_test,
    }


def _stochastic_sdae_payload(
    binding: StochasticProcessBinding,
    template: Any,
    *,
    seed: int,
    state_units: Mapping[str, str],
) -> Dict[str, Any]:
    """Build the stochastic_sdae_check context for the bound process graph.

    References:
        Revised RELIC foundations, stochastic and marked-jump semantics;
        Kloeden and Platen (1992); Gillespie (1977).
    Mathematical expression:
        continuous_sde: dX = a(X) dt + b dW on the declared term states;
        marked_jump: dN = R N(dt, dz) with R = [[1]] on the count state, an
        open ledger C_h R_h = [[1]] routed through the "port" vocabulary, and
        the mark law carried by mark_space.
    Parameter meanings:
        parameter_keys and propensity_keys are the ParameterManager keys of
        the Functor namespace; state_units come from the SSOT catalog.
    Units: as declared per state by the SSOT catalog.
    Domain/constraints: exactly one semantic per binding.
    Sign convention: not applicable.
    """
    # metadata.parameters declares the full ParameterManager keys.
    parameter_keys = list(template.metadata.parameters)
    payload: Dict[str, Any] = {
        "required_semantics": [binding.semantic],
        "runtime_owner": binding.solver_class.__module__,
        "stochastic_reproducibility": _reproducibility_map(binding, template, seed=seed),
        "state_units": dict(state_units),
        "execution_bindings": {binding.semantic: _execution_binding(binding)},
    }
    if binding.semantic == "continuous_sde":
        payload["stochastic_terms"] = [
            {
                "name": binding.term_name,
                "classification": template.noise_classification,
                "runtime_owner": binding.solver_owner,
                "state_vars": list(binding.state_variables),
                "parameter_keys": parameter_keys,
                "noise_type": binding.noise_type,
                "solver_name": template.solver_name,
            }
        ]
        return payload
    if binding.semantic == "marked_jump":
        count_state = binding.state_variables[0]
        propensity_key = binding.parameter_key("propensity")
        mark_key = binding.parameter_key("mark")
        payload["jump_resets"] = [
            {
                "name": binding.term_name,
                "reset_map": [[1.0]],
                "conservation_matrix": [[1.0]],
                "ledger_behavior": "open",
                "open_system_route": "port",
                "open_ledger_change": [[1.0]],
                "open_ledger_units": state_units[count_state],
                "propensity_keys": [propensity_key],
                "state_vars": list(binding.state_variables),
                "mark_space": (
                    f"{template.mark_law} depth mark z >= 0 "
                    f"[{state_units[binding.state_variables[1]]}] with mean {mark_key}"
                ),
                "event_clock": "direct_ssa_fixed_step",
            }
        ]
        payload["stochastic_reproducibility"]["jump_propensities"] = [propensity_key]
        return payload
    raise ValueError(f"Unsupported stochastic semantic {binding.semantic!r}.")


def build_bound_stochastic_graph(
    process: str,
    *,
    seed: int,
    semantic: Optional[str] = None,
) -> BoundStochasticGraph:
    """Build the canonical NodeRegistry and GraphIR of one bound stochastic process.

    References:
        Kloeden and Platen (1992), ch. 10; Higham (2001); Gillespie (1977);
        revised RELIC foundations, execution binding and claim-owner contracts.
    Mathematical expression:
        The graph carries one Functor node over its SCM states and the
        stochastic_sdae_check assertion that names that Functor, its solver
        owner, its states, and its executed validation test.
    Parameter meanings:
        process selects a BOUND_STOCHASTIC_PROCESSES entry (optionally
        restricted to one semantic); seed is the Functor-owned substream seed;
        coefficients come only from the ParameterManager database (GraphIR
        node config may not carry per-node parameter overrides).
    Units:
        State units are read from the SSOT catalog and recorded in the payload.
    Domain/constraints:
        seed is a non-negative integer below 2**63; unbound process names and
        untyped states fail closed.
    Sign convention:
        Not applicable (structural binding).
    """
    binding = require_bound_stochastic_process(process, semantic=semantic)
    config: Dict[str, Any] = {"seed": seed}
    template = binding.functor_class(name=binding.process, config=deepcopy(config))
    catalog = SSOTCatalog.from_parameter_manager()
    registry = NodeRegistry(catalog)
    definition = registry.register_functor_instance(template, node_type_id=binding.node_type)
    definition.metadata["config_contract"] = deepcopy(config)
    state_units: Dict[str, str] = {}
    for state in binding.state_variables:
        entry = catalog.get(state)
        unit = str(entry.units or "").strip() if entry is not None else ""
        if not unit:
            raise ValueError(
                f"SSOT catalog declares no units for the bound state {state!r}; the "
                "public stochastic binding requires typed state units."
            )
        state_units[state] = unit
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            binding.process: NodeSpec(
                class_type=binding.node_type,
                inputs={state: {"state": state} for state in binding.state_variables},
                outputs=list(template.metadata.output_variables),
                config=deepcopy(config),
            )
        },
        links=[],
        metadata={"stochastic_extension": True},
        assertion={
            "phi": (
                f"{', '.join(binding.state_variables)} advanced only by the bound "
                f"{_SEMANTIC_LABELS[binding.semantic]} Functor {binding.functor_class_path}"
            ),
            "op": "stochastic_sdae_check",
            "scope": "graph",
            "context": _stochastic_sdae_payload(
                binding, template, seed=seed, state_units=state_units
            ),
            "audit_key": "stochastic_sdae_audit",
        },
    )
    return BoundStochasticGraph(
        binding=binding,
        registry=registry,
        graph=graph,
        template=template,
        state_units=state_units,
    )


def build_bound_continuous_sde_graph(process: str, *, seed: int) -> BoundStochasticGraph:
    return build_bound_stochastic_graph(process, seed=seed, semantic="continuous_sde")


def build_bound_marked_jump_graph(process: str, *, seed: int) -> BoundStochasticGraph:
    return build_bound_stochastic_graph(process, seed=seed, semantic="marked_jump")


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _require_finite_real(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not bool(torch.isfinite(torch.tensor(float(value), dtype=torch.float64)))
    ):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    if positive and float(value) <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return float(value)


def _reconstruct_trajectory(
    executor: GraphExecutor,
    state: str,
    initial: Tensor,
    final_state: Mapping[str, Tensor],
    *,
    steps: int,
) -> Tensor:
    """Rebuild (sample, time) from the recorded increments; reject any drift."""
    increments = executor.time_series.get(state, [])
    if len(increments) != steps:
        raise RuntimeError(
            f"Governed execution recorded {len(increments)} increments for {state!r}, "
            f"expected {steps}."
        )
    rows: List[Tensor] = [initial.clone()]
    current = initial.clone()
    for increment in increments:
        if not isinstance(increment, Tensor):
            raise RuntimeError(f"Recorded increment for {state!r} is not a tensor.")
        current = current + increment.to(dtype=current.dtype)
        rows.append(current.clone())
    committed = final_state[state].detach().cpu().to(torch.float64).reshape(current.shape)
    if not torch.equal(rows[-1], committed):
        raise RuntimeError(
            "Trajectory reconstructed from the recorded increments does not equal "
            f"the committed StateManager state for {state!r}."
        )
    return torch.stack(rows, dim=1)


def execute_bound_stochastic_process(
    process: str,
    *,
    initial_values: Optional[Mapping[str, Any]] = None,
    n_realizations: int,
    dt_seconds: float,
    n_steps: int,
    seed: int,
    semantic: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one bound stochastic process through the canonical graph.

    References:
        Hasselmann (1976); Kloeden and Platen (1992), ch. 10; Higham (2001);
        Gillespie (1977); Karlin and Taylor (1981); Rodriguez-Iturbe et al.
        (1999).
    Mathematical expression:
        The governed Functor emits its increments for every realization; the
        trajectory of each state is rebuilt from the recorded increments and
        compared bitwise with the committed state.  Ensemble mean and unbiased
        variance are computed across realizations and the Functor's own
        discrete_moments supply the exact moments for comparison.
    Parameter meanings:
        initial_values per state (every bound state is required); n_realizations
        batched cells advanced by one Functor substream; dt_seconds the
        simulator step; n_steps fixed steps; seed the substream seed.
    Units:
        Per state as declared by the SSOT catalog; dt_seconds in s converted
        to day inside the Functor.
    Domain/constraints:
        n_realizations and n_steps positive integers; dt_seconds finite and
        positive; initial values finite and named by bound states only; the
        Functor's own domain checks apply to every step.
    Sign convention:
        As declared by the bound Functor.
    """
    realizations = _require_positive_integer(n_realizations, "n_realizations")
    steps = _require_positive_integer(n_steps, "n_steps")
    dt_value = _require_finite_real(dt_seconds, "dt_seconds", positive=True)
    bound = build_bound_stochastic_graph(process, seed=seed, semantic=semantic)
    states = bound.binding.state_variables
    provided = dict(initial_values or {})
    unknown = sorted(set(provided).difference(states))
    if unknown:
        raise ValueError(
            f"initial_values name states {unknown} that are not bound by {process!r}; "
            f"bound states: {list(states)}."
        )
    missing = [state for state in states if state not in provided]
    if missing:
        raise ValueError(
            f"initial_values missing for bound states {missing} of {process!r}; every "
            "bound state needs a declared initial value (no implicit zero)."
        )
    x0 = {state: _require_finite_real(provided[state], f"initial_values[{state}]") for state in states}

    plan = compile_graph(bound.graph, bound.registry, strict=True)
    executor = GraphExecutor(plan, registry=bound.registry)
    initial = {state: torch.full((realizations,), x0[state], dtype=torch.float64) for state in states}
    final_state = executor.execute(initial, n_steps=steps, dt=dt_value)
    trajectories = {
        state: _reconstruct_trajectory(executor, state, initial[state], final_state, steps=steps)
        for state in states
    }
    for state in states:
        spec = executor.state_manager.get_variable_spec(state)
        if spec is None or str(spec.units).strip() != bound.state_units[state]:
            raise RuntimeError(
                f"StateManager schema units for {state!r} do not match the SSOT catalog."
            )

    parameters = {
        name: float(bound.template.get_parameter_value(name, required=True).detach())
        for name in bound.binding.functor_class.PARAMETERS
    }
    moments = bound.binding.functor_class.discrete_moments(
        initial_values=x0,
        parameters=parameters,
        dt_days=dt_value / float(SECONDS_PER_DAY),
        n_steps=steps,
    )

    audit = plan.report.validation.assertion_audit
    structural = audit.get("structural_result") if isinstance(audit, Mapping) else None
    summary = executor.execution_summary
    result: Dict[str, Any] = {
        "process": bound.binding.process,
        "semantic": bound.binding.semantic,
        "functor_class": bound.binding.functor_class_path,
        "state_variables": list(states),
        "state_units": dict(bound.state_units),
        "seed": int(seed),
        "dt_seconds": dt_value,
        "n_steps": steps,
        "n_realizations": realizations,
        "initial_values": dict(x0),
        "parameter_keys": [bound.binding.parameter_key(name) for name in parameters],
        "parameters": parameters,
        "time_seconds": torch.arange(steps + 1, dtype=torch.float64) * dt_value,
        # Declarations read by relic.stochastic.io for the non-state tensors.
        "variable_units": {"time_seconds": "s"},
        "variable_dimensions": {"time_seconds": ["time"]},
        "assertion_audit": {
            "op": "stochastic_sdae_check",
            "present": bool(audit.get("present")) if isinstance(audit, Mapping) else False,
            "valid": bool(audit.get("valid")) if isinstance(audit, Mapping) else False,
            "op_resolved": bool(audit.get("op_resolved")) if isinstance(audit, Mapping) else False,
            "structural_passed": bool(structural.get("passed")) if isinstance(structural, Mapping) else False,
        },
        "execution": {
            "device": str(summary["device"]),
            "step_count": int(summary["step_count"]),
            "current_time": float(summary["current_time"]),
            "execution_order": ",".join(str(name) for name in summary["execution_order"]),
        },
    }
    for state in states:
        trajectory = trajectories[state]
        result[f"trajectory_{state}"] = trajectory
        result[f"ensemble_mean_{state}"] = trajectory.mean(dim=0)
        if realizations > 1:
            # Unbiased sample variance across realizations; omitted for a
            # single realization instead of reporting a degenerate zero.
            result[f"ensemble_variance_{state}"] = trajectory.var(dim=0, unbiased=True)
        result[f"analytic_mean_{state}"] = moments[state]["mean"]
        result[f"analytic_variance_{state}"] = moments[state]["variance"]
    return result


def execute_bound_continuous_sde(
    process: str,
    *,
    initial_value: float,
    n_realizations: int,
    dt_seconds: float,
    n_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Execute one bound continuous-SDE process (single prognostic state)."""
    binding = require_bound_continuous_sde_process(process)
    return execute_bound_stochastic_process(
        process,
        initial_values={binding.state_variables[0]: initial_value},
        n_realizations=n_realizations,
        dt_seconds=dt_seconds,
        n_steps=n_steps,
        seed=seed,
        semantic="continuous_sde",
    )


def execute_bound_marked_jump(
    process: str,
    *,
    n_realizations: int,
    dt_seconds: float,
    n_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Execute one bound marked-jump process from zero counts and depths."""
    # The marked-jump states are event counters and accumulated depths of the
    # observation window; the zero start is declared here by this owner.
    binding = require_bound_marked_jump_process(process)
    return execute_bound_stochastic_process(
        process,
        initial_values={state: 0.0 for state in binding.state_variables},
        n_realizations=n_realizations,
        dt_seconds=dt_seconds,
        n_steps=n_steps,
        seed=seed,
        semantic="marked_jump",
    )

def run_sde_simulation(args) -> None:
    """Execute the CLI SDE command through the bound process graph.

    References:
        Kloeden and Platen (1992); revised foundations, continuous SDE and
        claim-owner-evidence contracts.
    Mathematical expression:
        dX = a(X,t) dt + b(X,t) dW executed by the bound Functor selected with
        --process; see execute_bound_continuous_sde for the moments reported.
    Parameter meanings:
        --process bound process name; --samples realizations; --steps fixed
        steps; --dt step in seconds; --initial-value X_0; --seed substream
        seed; the optional config section sde may supply the same fields.
    Units:
        dt in s; X_0 and outputs in the SCM state unit.
    Domain/constraints:
        The seed is mandatory; unbound process names, per-run parameter
        overrides (parameters belong to the ParameterManager database), and
        invalid requests fail before any output is written.
    Sign convention:
        As declared by the bound Functor.
    """
    config = load_config(args.config) if getattr(args, "config", None) else {}
    section = get_section(config, "sde")

    def _select(attribute: str, key: str) -> Any:
        value = getattr(args, attribute, None)
        return section.get(key) if value is None else value

    seed = _select("seed", "seed")
    if seed is None:
        raise ValueError(
            "SDE execution requires an integer seed (--seed or sde.seed) for the "
            "Functor-owned Wiener substream."
        )
    if "parameters" in section:
        raise ValueError(
            "sde.parameters is not accepted: drift and diffusion coefficients are "
            "owned by the ParameterManager database (stochastic.weather.temperature.*)."
        )
    result = execute_bound_continuous_sde(
        _select("process", "process"),
        initial_value=_select("initial_value", "initial_value"),
        n_realizations=_select("samples", "samples"),
        dt_seconds=_select("dt", "dt_seconds"),
        n_steps=_select("steps", "steps"),
        seed=seed,
    )
    output_path = getattr(args, "output", None) or "sde_results"
    save_results(result, output_path, getattr(args, "format", "netcdf"))
    logger.info(
        "Bound SDE execution completed: process=%s realizations=%s steps=%s",
        result["process"],
        result["n_realizations"],
        result["n_steps"],
    )


def run_gillespie_simulation(args) -> None:
    """Execute the CLI marked-jump command through the bound process graph.

    References:
        Gillespie (1977); Rodriguez-Iturbe et al. (1999); revised foundations,
        marked-jump and claim-owner-evidence contracts.
    Mathematical expression:
        dN = R N(dt, dz) executed by the bound Functor selected with
        --process from zero counts and depths; see
        execute_bound_stochastic_process for the moments reported.
    Parameter meanings:
        --process bound marked-jump process name; --samples realizations;
        --steps fixed steps; --dt step in seconds; --seed substream seed; the
        optional config section gillespie may supply the same fields.
    Units:
        dt in s; outputs in the SCM state units.
    Domain/constraints:
        The seed is mandatory; unbound process names, arbitrary reaction
        declarations, per-run parameter overrides, and invalid requests fail
        before any output is written.
    Sign convention:
        As declared by the bound Functor.
    """
    config = load_config(args.config) if getattr(args, "config", None) else {}
    section = get_section(config, "gillespie")

    def _select(attribute: str, key: str) -> Any:
        value = getattr(args, attribute, None)
        return section.get(key) if value is None else value

    seed = _select("seed", "seed")
    if seed is None:
        raise ValueError(
            "Marked-jump execution requires an integer seed (--seed or gillespie.seed) "
            "for the Functor-owned event substream."
        )
    for forbidden in ("parameters", "reactions", "species"):
        if forbidden in section:
            raise ValueError(
                f"gillespie.{forbidden} is not accepted: arbitrary reaction networks are "
                "[N] and coefficients are owned by the ParameterManager database."
            )
    result = execute_bound_marked_jump(
        _select("process", "process"),
        n_realizations=_select("samples", "samples"),
        dt_seconds=_select("dt", "dt_seconds"),
        n_steps=_select("steps", "steps"),
        seed=seed,
    )
    output_path = getattr(args, "output", None) or "gillespie_results"
    save_results(result, output_path, getattr(args, "format", "netcdf"))
    logger.info(
        "Bound marked-jump execution completed: process=%s realizations=%s steps=%s",
        result["process"],
        result["n_realizations"],
        result["n_steps"],
    )

def run_validation_analysis(args) -> None:
    """Run validation analysis for stochastic results."""
    try:
        if not args.results:
            raise ValueError("Results file required for validation")

        results_path = Path(args.results)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {args.results}")

        result_dimensions: Dict[str, Tuple[str, ...]] = {}
        if results_path.suffix == ".nc":
            with xr.open_dataset(results_path) as dataset:
                results = {
                    str(var): torch.tensor(dataset[var].values)
                    for var in dataset.data_vars
                }
                result_dimensions = {
                    str(var): tuple(str(dim) for dim in dataset[var].dims)
                    for var in dataset.data_vars
                }
        else:
            results = torch.load(
                results_path,
                map_location="cpu",
                weights_only=True,
            )

        # PIT, CRPS and coverage compare the ensemble with observations; a
        # validation run without them has nothing to compute and raises
        # instead of writing an empty result.
        if not getattr(args, "observations", None):
            raise ValueError(
                "Validation analysis needs an observations file (--observations)"
            )
        obs_path = Path(args.observations)
        if obs_path.suffix == ".nc":
            with xr.open_dataset(obs_path) as obs_dataset:
                observations = {var: torch.tensor(obs_dataset[var].values) for var in obs_dataset.data_vars}
        else:
            observations = torch.load(
                obs_path,
                map_location="cpu",
                weights_only=True,
            )

        # The nominal coverage is the stated value or, when None, the
        # parameter-database value (its single source); a namespace without
        # the attribute raises AttributeError (no second default here).
        declared_coverage = args.nominal_coverage
        if declared_coverage is None:
            declared_coverage = get_global_manager().get(
                "validation.stochastic_scores.nominal_interval_coverage", required=True
            )
        nominal_coverage_tensor = torch.as_tensor(
            declared_coverage,
            dtype=torch.float64,
        )
        if nominal_coverage_tensor.ndim != 0:
            raise ValueError("nominal_coverage must be a scalar")
        nominal_coverage = float(nominal_coverage_tensor.item())
        if (
            not bool(torch.isfinite(nominal_coverage_tensor).item())
            or not 0.0 < nominal_coverage < 1.0
        ):
            raise ValueError(
                "nominal_coverage must be finite and strictly between 0 and 1"
            )
        if not isinstance(results, dict):
            raise TypeError("Validation results file must contain a mapping")
        if not isinstance(observations, dict):
            raise TypeError("Validation observations file must contain a mapping")

        matched_keys = [key for key in results if key in observations]
        if not matched_keys:
            raise KeyError(
                f"No result variable has observations: results {sorted(results)}, "
                f"observations {sorted(observations)}"
            )
        unobserved = sorted(key for key in results if key not in observations)
        if unobserved:
            logger.info("Result variables without observations are not validated: %s", unobserved)

        validation_results: Dict[str, Any] = {}
        # References:
        #   Gneiting and Raftery (2007); Wilks (2011).
        # Formula:
        #   PIT = F(y_obs); CRPS = integral (F(z) - H(z-y))^2 dz;
        #   alpha = (1-c)/2, L = Q_alpha(X), U = Q_(1-alpha)(X).
        # Parameters:
        #   observations y_obs, ensemble X, nominal coverage c.
        for key, data in results.items():
            if key in observations:
                obs_data, ensemble = _prepare_validation_ensemble(
                    observations[key],
                    data,
                    result_dimensions.get(key),
                )
                tail_probability = (1.0 - nominal_coverage) / 2.0
                quantile_levels = torch.tensor(
                    [tail_probability, 1.0 - tail_probability],
                    dtype=ensemble.dtype,
                    device=ensemble.device,
                )
                lower_bounds, upper_bounds = torch.quantile(
                    ensemble,
                    quantile_levels,
                    dim=1,
                )
                pit_result = pit_test(obs_data, ensemble)
                crps_result = crps_score(obs_data, ensemble)
                coverage_result = coverage_probability(
                    obs_data,
                    lower_bounds,
                    upper_bounds,
                    nominal_coverage,
                )
                validation_results[key] = _summarize_validation_results(
                    pit_result,
                    crps_result,
                    coverage_result,
                    nominal_coverage,
                )

        output_path = args.output or "validation_results"
        save_results(validation_results, output_path, args.format)

        logger.info("Validation analysis completed")

    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
        logger.error("Validation analysis failed: %s", exc)
        raise
