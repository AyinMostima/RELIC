"""
Stochastic process API routes.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from relic.stochastic import get_available_samplers
from relic.stochastic.workflows import (
    BOUND_CONTINUOUS_SDE_PROCESSES,
    BOUND_MARKED_JUMP_PROCESSES,
    execute_bound_continuous_sde,
    execute_bound_marked_jump,
)

from ..schemas import (
    APIResponse,
    MarkedJumpRequest,
    SDERequest,
    WeatherGenerationRequest,
    create_error_response,
    create_success_response,
)

router = APIRouter()

_BOUND_RESULT_SCALAR_KEYS = (
    "process",
    "semantic",
    "functor_class",
    "state_variables",
    "state_units",
    "seed",
    "dt_seconds",
    "n_steps",
    "n_realizations",
    "initial_values",
    "parameter_keys",
    "parameters",
    "assertion_audit",
    "execution",
)


def _serialize_bound_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a bound stochastic execution result for the JSON response."""
    data = {key: result[key] for key in _BOUND_RESULT_SCALAR_KEYS}
    data["trajectories"] = {
        state: result[f"trajectory_{state}"].tolist() for state in result["state_variables"]
    }
    states: Dict[str, Dict[str, Any]] = {}
    for state in result["state_variables"]:
        variance = result.get(f"ensemble_variance_{state}")
        states[state] = {
            "ensemble_mean": result[f"ensemble_mean_{state}"].tolist(),
            "ensemble_variance": variance.tolist() if variance is not None else None,
            "analytic_mean": result[f"analytic_mean_{state}"].tolist(),
            "analytic_variance": result[f"analytic_variance_{state}"].tolist(),
        }
    data["statistics"] = {"time_seconds": result["time_seconds"].tolist(), "states": states}
    return data


@router.get("/", response_model=APIResponse)  # type: ignore[misc]
async def stochastic_info() -> APIResponse:
    samplers = get_available_samplers()
    weather_contract = samplers["WeatherGenerator"]
    sde_contract = samplers["EulerMaruyamaSolver"]
    gillespie_contract = samplers["GillespieSSA"]
    return create_success_response(
        message="Stochastic endpoints are available.",
        data={
            "status": "partial",
            "endpoints": ["weather", "sde", "gillespie"],
            "weather": {
                "status": weather_contract["status"],
                "runtime_ready": weather_contract["runtime_ready"],
                "reason": (
                    "A declared weather-forcing operator does not yet have a "
                    "validated GraphIR/Functor runtime binding."
                ),
                "required_evidence": list(weather_contract["requirements"]),
            },
            "sde": {
                "status": sde_contract["status"],
                "runtime_ready": sde_contract["runtime_ready"],
                "bound_processes": list(sde_contract["bound_processes"]),
                "graph_owner": sde_contract["graph_owner"],
                "reason": (
                    "Bound processes execute through the canonical GraphIR/"
                    "Functor path with a Functor-owned Wiener substream; "
                    "unbound process names fail closed."
                ),
            },
            "gillespie": {
                "status": gillespie_contract["status"],
                "runtime_ready": gillespie_contract["runtime_ready"],
                "bound_processes": list(gillespie_contract["bound_processes"]),
                "graph_owner": gillespie_contract["graph_owner"],
                "reason": (
                    "Bound marked-jump processes execute through the canonical "
                    "GraphIR/Functor path with a Functor-owned event substream; "
                    "unbound process names and arbitrary reaction-network "
                    "declarations fail closed."
                ),
            },
            "supported_sde_processes": list(sde_contract["bound_processes"]),
            "supported_marked_jump_processes": list(gillespie_contract["bound_processes"]),
            "weather_variables": ["air_temperature", "precipitation_flux"],
        },
    )


@router.post(  # type: ignore[misc]
    "/weather",
    response_model=APIResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Weather forcing runtime binding [N]",
    description=(
        "This endpoint is intentionally fail-closed until a declared weather-"
        "forcing operator has a validated GraphIR/Functor runtime binding."
    ),
)
async def generate_weather(request: WeatherGenerationRequest) -> APIResponse:
    """Reject weather forcing execution without canonical runtime evidence.

    References:
        Hasselmann (1976); Wilks (2006); RELIC Mathematical Foundations
        Revised v2, stochastic forcing and runtime-binding contracts.
    Mathematical expression:
        dT = a(T,t) dt + b(T,t) dW and marked precipitation events may run
        only when their operators and stochastic path are explicitly bound.
    Parameter meanings:
        a and b are declared temperature operators; event intensities and
        marks define precipitation under canonical parameter, unit, calendar,
        state, forcing-cursor, checkpoint, and RNG contracts.
    """
    del request
    contract = get_available_samplers()["WeatherGenerator"]
    message = (
        "Weather generation requires a declared weather-forcing operator and "
        "a validated GraphIR/Functor runtime binding. This integration is "
        "[N]; the API will not infer a scientific forcing law."
    )
    return create_error_response(
        message=message,
        error_type="WeatherGenerationValidationError",
        details={
            "status": contract["status"],
            "required_evidence": list(contract["requirements"]),
        },
    )


@router.post(  # type: ignore[misc]
    "/sde",
    response_model=APIResponse,
    summary="Execute a bound continuous-SDE process",
    description=(
        "Executes one bound continuous-SDE scientific Functor through the "
        "canonical GraphIR/Functor path (compile_graph strict, GraphExecutor, "
        "StateManager commit) and returns the governed trajectory with its "
        "ensemble and analytic conditional moments. Unbound process names and "
        "per-run parameter overrides fail closed."
    ),
)
async def solve_sde(request: SDERequest) -> APIResponse:
    """Execute a bound continuous-SDE process through the canonical graph.

    References:
        Hasselmann (1976); Kloeden and Platen (1992), ch. 10; Higham (2001);
        RELIC Mathematical Foundations Revised v2, continuous-SDE and
        runtime-binding contracts.
    Mathematical expression:
        dX_t = a(X_t, t) dt + B(X_t, t) dW_t executed by the bound Functor as
        X_{n+1} = X_n + a(X_n) dt + B dW_n; the response reports the governed
        trajectory, ensemble mean/variance, and the conditional AR(1) moments
        of the discrete recursion.
    Parameter meanings:
        process selects the bound Functor; initial_value X_0; dt_seconds the
        fixed simulator step; n_steps fixed steps; n_realizations batched
        cells on one Functor substream; seed the substream seed.
    Units:
        dt_seconds in s; X and moments in the SCM state unit reported as
        state_unit; parameters in their ParameterManager units.
    Domain/constraints:
        The bound Functor rejects inadmissible steps (alpha dt outside (0, 1))
        and non-finite states; validation failures return HTTP 422.
    Sign convention:
        As declared by the bound Functor (positive increment warms the anomaly).
    """
    try:
        result = execute_bound_continuous_sde(
            request.process,
            initial_value=request.initial_value,
            n_realizations=request.n_realizations,
            dt_seconds=request.dt_seconds,
            n_steps=request.n_steps,
            seed=request.seed,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "type": "SDEValidationError",
                "message": str(exc),
                "bound_processes": sorted(BOUND_CONTINUOUS_SDE_PROCESSES),
            },
        ) from exc

    data = _serialize_bound_result(result)
    return create_success_response(
        message=(
            f"Bound continuous-SDE process {result['process']} executed "
            f"{result['n_steps']} governed steps for {result['n_realizations']} "
            "realizations."
        ),
        data=data,
    )


@router.post(  # type: ignore[misc]
    "/gillespie",
    response_model=APIResponse,
    summary="Execute a bound marked-jump process",
    description=(
        "Executes one bound marked-jump scientific Functor (direct-SSA event "
        "clock with the Functor's mark law) through the canonical GraphIR/"
        "Functor path and returns the governed count and mark trajectories "
        "with their ensemble and analytic moments. Unbound process names and "
        "arbitrary reaction-network declarations fail closed."
    ),
)
async def run_gillespie(request: MarkedJumpRequest) -> APIResponse:
    """Execute a bound marked-jump process through the canonical graph.

    References:
        Gillespie (1977), J. Phys. Chem. 81, 2340-2361; Rodriguez-Iturbe et
        al. (1999); RELIC Mathematical Foundations Revised v2, marked-jump
        and runtime-binding contracts.
    Mathematical expression:
        dX = h(X^-, z) N(dt, dz) executed by the bound Functor: the direct SSA
        samples the arrivals of each fixed step and the Functor applies its
        mark law; the response reports trajectories, ensemble mean/variance,
        and the exact compound-Poisson moments.
    Parameter meanings:
        process selects the bound Functor; dt_seconds the fixed simulator
        step; n_steps fixed steps; n_realizations batched cells on one
        Functor substream; seed the substream seed.
    Units:
        dt_seconds in s; states in the SCM units reported as state_units.
    Domain/constraints:
        Counts and marks start at zero; validation failures return HTTP 422.
    Sign convention:
        As declared by the bound Functor (nonnegative arrivals and marks).
    """
    try:
        result = execute_bound_marked_jump(
            request.process,
            n_realizations=request.n_realizations,
            dt_seconds=request.dt_seconds,
            n_steps=request.n_steps,
            seed=request.seed,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "type": "GillespieValidationError",
                "message": str(exc),
                "bound_processes": sorted(BOUND_MARKED_JUMP_PROCESSES),
            },
        ) from exc

    return create_success_response(
        message=(
            f"Bound marked-jump process {result['process']} executed "
            f"{result['n_steps']} governed steps for {result['n_realizations']} "
            "realizations."
        ),
        data=_serialize_bound_result(result),
    )
