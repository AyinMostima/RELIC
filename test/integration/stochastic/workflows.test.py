"""L2 tests for stochastic validation workflow wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import xarray as xr
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from relic.interfaces.api.routes import stochastic as stochastic_api
from relic.interfaces.cli.stochastic import execute
from relic.interfaces.api.routes.stochastic import (
    generate_weather,
    router as stochastic_router,
    solve_sde,
    stochastic_info,
)
from relic.interfaces.api.schemas import SDERequest, WeatherGenerationRequest
from relic.scientific.atmosphere.stochastic_rainfall import MarkedPoissonRainfallFunctor
from relic.scientific.atmosphere.stochastic_temperature import (
    OrnsteinUhlenbeckTemperatureFunctor,
)
from relic.stochastic import get_available_samplers
from relic.stochastic import workflows
from relic.stochastic.sde_solver import EulerMaruyamaSolver, SDEParameters

BOUND_SDE_PROCESS = "stochastic_air_temperature"
BOUND_SDE_STATE = "stochastic_air_temperature_anomaly"
BOUND_JUMP_PROCESS = "stochastic_rainfall"
RAIN_COUNT = "stochastic_rainfall_event_count"
RAIN_DEPTH = "stochastic_rainfall_depth"


def _reference_rainfall_trajectories(*, seed, n_realizations, dt_seconds, n_steps, device):
    """Standalone same-seed Functor replay of the governed marked-jump path."""
    functor = MarkedPoissonRainfallFunctor(name=BOUND_JUMP_PROCESS, config={"seed": seed}, device=device)
    values = {
        RAIN_COUNT: torch.zeros(n_realizations, dtype=torch.float64, device=device),
        RAIN_DEPTH: torch.zeros(n_realizations, dtype=torch.float64, device=device),
    }
    rows = {name: [value.detach().cpu().clone()] for name, value in values.items()}
    for step in range(n_steps):
        outputs = functor.forward(dict(values), step * dt_seconds, dt_seconds)
        for name in values:
            values[name] = (values[name] + outputs[name].value).detach()
            rows[name].append(values[name].cpu().clone())
    return {name: torch.stack(items, dim=1) for name, items in rows.items()}


def _reference_ou_trajectory(
    *,
    seed: int,
    initial_value: float,
    n_realizations: int,
    dt_seconds: float,
    n_steps: int,
    device,
    parameters=None,
) -> torch.Tensor:
    """Standalone same-seed Functor replay of the governed path (sample, time)."""
    config = {"seed": seed}
    if parameters:
        config["parameters"] = dict(parameters)
    functor = OrnsteinUhlenbeckTemperatureFunctor(
        name=BOUND_SDE_PROCESS, config=config, device=device
    )
    value = torch.full((n_realizations,), float(initial_value), dtype=torch.float64, device=device)
    rows = [value.detach().cpu().clone()]
    for step in range(n_steps):
        increment = functor.forward(
            {BOUND_SDE_STATE: value}, step * dt_seconds, dt_seconds
        )[BOUND_SDE_STATE].value
        value = (value + increment).detach()
        rows.append(value.cpu().clone())
    return torch.stack(rows, dim=1)

pytestmark = pytest.mark.integration


def _zero_drift(
    state: torch.Tensor,
    time: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    del time, kwargs
    return torch.zeros_like(state)


def _constant_diffusion(
    state: torch.Tensor,
    time: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    del time, kwargs
    return torch.full_like(state, 0.2)


def test_sde_solver_requires_explicit_drift_and_diffusion_operators() -> None:
    with pytest.raises(ValueError, match="explicit drift_func and diffusion_func"):
        EulerMaruyamaSolver(parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"))


def test_sde_solver_rejects_unimplemented_adaptive_wiener_semantics() -> None:
    # The standalone solver stays fixed-step; variable steps on one Wiener
    # path belong to the simulator adaptive control with the Functor-owned
    # Brownian tree (F05).
    with pytest.raises(ValueError, match="Brownian tree"):
        EulerMaruyamaSolver(
            parameters=SDEParameters(dt=0.1, adaptive=True, method="euler_maruyama", diffusion_structure="diagonal"),
            drift_func=_zero_drift,
            diffusion_func=_constant_diffusion,
        )


def test_sde_solver_exposes_only_the_supported_fixed_euler_contract() -> None:
    with pytest.raises(TypeError, match="required positional argument: 'dt'"):
        SDEParameters()

    for method in ("milstein", "predictor_corrector"):
        with pytest.raises(ValueError, match="not operational"):
            EulerMaruyamaSolver(
                parameters=SDEParameters(dt=0.1, method=method, diffusion_structure="diagonal"),
                drift_func=_zero_drift,
                diffusion_func=_constant_diffusion,
            )

    with pytest.raises(ValueError, match="dt"):
        SDEParameters(dt=True, method="euler_maruyama", diffusion_structure="diagonal")


def test_sde_solver_is_numeric_only_and_uses_a_local_seeded_stream() -> None:
    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=_zero_drift,
        diffusion_func=_constant_diffusion,
    )
    torch.manual_seed(1234)
    global_rng_before = torch.random.get_rng_state().clone()

    first = solver.sample(
        torch.tensor([1.0]),
        num_samples=2,
        seed=99,
        num_steps=3,
    )
    second = solver.sample(
        torch.tensor([1.0]),
        num_samples=2,
        seed=99,
        num_steps=3,
    )

    assert torch.equal(first["solution"], second["solution"])
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert not hasattr(solver, "process_params")
    assert not hasattr(solver, "set_process_parameters")
    torch.testing.assert_close(
        first["time"],
        torch.tensor([0.0, 0.1, 0.2, 0.3]),
    )


def test_sde_solver_external_generator_supports_state_resume() -> None:
    def time_dependent_drift(state, time, **kwargs):
        del kwargs
        return torch.ones_like(state) * time

    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=time_dependent_drift,
        diffusion_func=_constant_diffusion,
    )
    initial_state = torch.tensor([1.0])

    full_generator = torch.Generator().manual_seed(17)
    full = solver.sample(
        initial_state,
        num_samples=1,
        num_steps=4,
        generator=full_generator,
    )

    split_generator = torch.Generator().manual_seed(17)
    first = solver.sample(
        initial_state,
        num_samples=1,
        num_steps=2,
        generator=split_generator,
    )
    saved_generator_state = split_generator.get_state().clone()
    resumed_generator = torch.Generator()
    resumed_generator.set_state(saved_generator_state)
    second = solver.sample(
        first["solution"][0, -1],
        num_samples=1,
        num_steps=2,
        generator=resumed_generator,
        initial_time=first["time"][-1],
    )
    resumed_path = torch.cat(
        (first["solution"], second["solution"][:, 1:]),
        dim=1,
    )

    torch.testing.assert_close(resumed_path, full["solution"])
    torch.testing.assert_close(second["time"][0], first["time"][-1])
    assert torch.equal(resumed_generator.get_state(), full_generator.get_state())


def test_sde_solver_rejects_seed_and_generator_together() -> None:
    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=_zero_drift,
        diffusion_func=_constant_diffusion,
    )

    with pytest.raises(ValueError, match="seed and generator are mutually exclusive"):
        solver.sample(
            torch.tensor([1.0]),
            seed=7,
            generator=torch.Generator().manual_seed(7),
        )


def test_sde_solver_evaluates_batched_operators_once_per_step() -> None:
    calls = {"drift": 0, "diffusion": 0}

    def drift(state, time, **kwargs):
        del time, kwargs
        calls["drift"] += 1
        assert state.shape == (4, 2)
        return torch.zeros_like(state)

    def diffusion(state, time, **kwargs):
        del time, kwargs
        calls["diffusion"] += 1
        assert state.shape == (4, 2)
        return torch.full_like(state, 0.1)

    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=drift,
        diffusion_func=diffusion,
    )
    solver.sample(
        torch.tensor([1.0, 2.0]),
        num_samples=4,
        num_steps=3,
        seed=11,
    )

    assert calls == {"drift": 3, "diffusion": 3}


def test_sde_solver_declares_only_diagonal_diffusion() -> None:
    with pytest.raises(ValueError, match="diffusion_structure"):
        SDEParameters(dt=0.1, diffusion_structure="matrix", method="euler_maruyama")

    parameters = SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal")
    assert parameters.diffusion_structure == "diagonal"


def test_sde_solver_does_not_silently_clip_negative_states() -> None:
    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=1.0, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=lambda state, time, **kwargs: torch.full_like(state, -1.0),
        diffusion_func=_zero_drift,
    )

    result = solver.sample(
        torch.tensor([0.1]),
        num_samples=1,
        num_steps=1,
        seed=5,
    )

    torch.testing.assert_close(
        result["solution"][0, -1],
        torch.tensor([-0.9]),
    )


def test_sde_solver_category_contract_is_initialized_and_unproved_statistics_fail_closed() -> None:
    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=_zero_drift,
        diffusion_func=_constant_diffusion,
    )

    description = solver.describe()
    assert description["name"] == "EulerMaruyamaSolver"
    assert description["parameters"]["method"] == "euler_maruyama"
    assert description["parameters"]["diffusion_structure"] == "diagonal"
    assert description["cf_compliant"] is False
    assert description["cf_validation_scope"] == "not_evaluated"
    assert description["dataset_conformance_checked"] is False
    assert description["unit_binding_status"] == "[N]"
    assert solver.metadata.name == "EulerMaruyamaSolver"

    with pytest.raises(NotImplementedError, match="transition-density"):
        solver.log_probability(torch.tensor([1.0]), torch.tensor([1.1]))
    with pytest.raises(NotImplementedError, match="analytical moments"):
        solver.moments(torch.tensor([1.0]))


@pytest.mark.parametrize(
    ("argument", "value"),
    [("num_samples", 1.5), ("num_steps", 2.5), ("seed", True)],
)
def test_sde_solver_rejects_nonintegral_path_controls(argument, value) -> None:
    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=_zero_drift,
        diffusion_func=_constant_diffusion,
    )
    kwargs = {"num_samples": 1, "num_steps": 2, "seed": 9}
    kwargs[argument] = value

    with pytest.raises(ValueError, match=argument):
        solver.sample(torch.tensor([1.0]), **kwargs)


def test_sde_solver_multistep_path_preserves_autograd() -> None:
    drift_scale = torch.tensor(0.2, requires_grad=True)

    def differentiable_drift(state, time, **kwargs):
        del time, kwargs
        return drift_scale * state

    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=0.1, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=differentiable_drift,
        diffusion_func=_zero_drift,
    )
    initial_state = torch.tensor([1.0], requires_grad=True)

    result = solver.sample(initial_state, num_samples=1, seed=7, num_steps=3)
    final_state = result["solution"][0, -1].sum()
    final_state.backward()

    expected_factor = (1.0 + 0.1 * drift_scale.detach()) ** 3
    torch.testing.assert_close(initial_state.grad, expected_factor.reshape(1))
    assert drift_scale.grad is not None
    assert torch.isfinite(drift_scale.grad)


def test_sde_solver_rejects_nonfinite_updated_state() -> None:
    solver = EulerMaruyamaSolver(
        parameters=SDEParameters(dt=10.0, method="euler_maruyama", diffusion_structure="diagonal"),
        drift_func=lambda state, time, **kwargs: state,
        diffusion_func=_zero_drift,
    )

    with pytest.raises(ValueError, match="Euler-Maruyama update produced"):
        solver.sample(
            torch.tensor([1.0e38], dtype=torch.float32),
            num_samples=1,
            seed=7,
            num_steps=1,
        )


def test_sde_sampler_health_reports_the_bound_continuous_sde_process() -> None:
    metadata = get_available_samplers()["EulerMaruyamaSolver"]

    assert metadata["available"] is True
    assert metadata["runtime_ready"] is True
    assert metadata["status"] == "bound"
    assert metadata["contract"] == "numeric_only"
    assert metadata["bound_processes"] == [BOUND_SDE_PROCESS]
    assert metadata["graph_owner"] == (
        "relic.stochastic.workflows.build_bound_continuous_sde_graph"
    )
    assert set(metadata["requirements"]) >= {
        "declared drift operator",
        "declared diffusion operator",
        "validated GraphIR/Functor runtime binding",
        "restartable stochastic substream binding",
    }


def test_weather_sampler_health_distinguishes_numeric_class_from_runtime_binding() -> None:
    metadata = get_available_samplers()["WeatherGenerator"]

    assert metadata["available"] is True
    assert metadata["contract"] == "numeric_only"
    assert metadata["runtime_ready"] is False
    assert metadata["status"] == "[N]"
    assert set(metadata["requirements"]) >= {
        "declared weather-forcing operator",
        "validated GraphIR/Functor runtime binding",
        "StateManager forcing/state binding",
        "ParameterManager and unit binding",
        "simulator calendar and forcing-cursor binding",
        "durable simulator checkpoint binding",
        "restartable stochastic substream binding",
    }


def test_gillespie_sampler_health_reports_the_bound_marked_jump_process() -> None:
    metadata = get_available_samplers()["GillespieSSA"]

    assert metadata["available"] is True
    assert metadata["contract"] == "numeric_only"
    assert metadata["runtime_ready"] is True
    assert metadata["status"] == "bound"
    assert metadata["bound_processes"] == [BOUND_JUMP_PROCESS]
    assert metadata["graph_owner"] == "relic.stochastic.workflows.build_bound_marked_jump_graph"
    assert set(metadata["requirements"]) >= {
        "declared jump/reaction operator",
        "declared propensity and stoichiometric reset map",
        "conservation matrix or declared open-system route",
        "validated GraphIR/Functor runtime binding",
        "StateManager initial-state binding",
        "ParameterManager and unit binding",
        "restartable stochastic substream binding",
    }


def test_cli_weather_workflow_fails_closed_before_config_sampling_or_output(
    monkeypatch,
    tmp_path,
) -> None:
    args = SimpleNamespace(
        config=str(tmp_path / "unbound_weather.yaml"),
        samples=2,
        duration=3,
        correlation=0.0,
        seed=7,
        output=str(tmp_path / "weather"),
        format="json",
    )
    for name in (
        "load_config",
        "_build_temperature_parameters",
        "_build_precipitation_parameters",
        "WeatherGenerator",
        "save_results",
    ):
        monkeypatch.setattr(
            workflows,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"fail-closed path invoked {_name}"
            ),
            raising=False,
        )

    with pytest.raises(
        ValueError,
        match=r"weather generation.*GraphIR/Functor runtime binding.*\[N\]",
    ):
        workflows.run_weather_generation(args)


def _sde_cli_args(tmp_path, **overrides) -> SimpleNamespace:
    values = {
        "config": None,
        "process": BOUND_SDE_PROCESS,
        "samples": 3,
        "steps": 4,
        "dt": 3600.0,
        "initial_value": 1.0,
        "seed": 7,
        "output": str(tmp_path / "sde"),
        "format": "json",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_sde_workflow_executes_the_bound_graph_and_writes_results(tmp_path) -> None:
    args = _sde_cli_args(tmp_path)

    workflows.run_sde_simulation(args)

    with open(tmp_path / "sde.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    trajectory = torch.tensor(payload[f"trajectory_{BOUND_SDE_STATE}"], dtype=torch.float64)
    assert trajectory.shape == (3, 5)
    assert payload["process"] == BOUND_SDE_PROCESS
    assert payload["seed"] == 7
    assert payload["time_seconds"] == [0.0, 3600.0, 7200.0, 10800.0, 14400.0]
    assert len(payload[f"ensemble_mean_{BOUND_SDE_STATE}"]) == 5
    assert len(payload[f"ensemble_variance_{BOUND_SDE_STATE}"]) == 5
    assert len(payload[f"analytic_mean_{BOUND_SDE_STATE}"]) == 5
    assert payload["state_units"] == {BOUND_SDE_STATE: "K"}
    assert payload["semantic"] == "continuous_sde"
    assert payload["assertion_audit"]["structural_passed"] is True
    reference = _reference_ou_trajectory(
        seed=7,
        initial_value=1.0,
        n_realizations=3,
        dt_seconds=3600.0,
        n_steps=4,
        device=payload["execution"]["device"],
    )
    torch.testing.assert_close(trajectory, reference, rtol=0.0, atol=0.0)


def test_cli_sde_workflow_writes_a_cf_netcdf_dataset_with_a_physical_time_axis(tmp_path) -> None:
    args = _sde_cli_args(tmp_path, samples=2, steps=3, format="netcdf")

    workflows.run_sde_simulation(args)

    with xr.open_dataset(tmp_path / "sde.nc") as dataset:
        trajectory = dataset[f"trajectory_{BOUND_SDE_STATE}"]
        assert trajectory.dims == ("sample", "time")
        assert trajectory.shape == (2, 4)
        # Units come from the SSOT-declared state units; the SSOT catalog
        # declares no CF standard name for the stochastic state, so none is
        # guessed from the variable name.
        assert "standard_name" not in trajectory.attrs
        assert trajectory.attrs["units"] == "K"
        assert "time" not in dataset.coords
        assert dataset["time_seconds"].dims == ("time",)
        assert dataset["time_seconds"].attrs["units"] == "s"
        assert dataset["time_seconds"].values.tolist() == [0.0, 3600.0, 7200.0, 10800.0]
        assert dataset[f"ensemble_variance_{BOUND_SDE_STATE}"].attrs["units"] == "K2"
        assert dataset.attrs["process"] == BOUND_SDE_PROCESS
        assert dataset.attrs[f"state_units_{BOUND_SDE_STATE}"] == "K"
        assert int(dataset["execution_step_count"].values) == 3
        assert int(dataset["seed"].values) == 7


def test_cli_sde_workflow_requires_a_seed_and_a_bound_process(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workflows,
        "save_results",
        lambda *args, **kwargs: pytest.fail("rejected request wrote output"),
    )
    with pytest.raises(ValueError, match="seed"):
        workflows.run_sde_simulation(_sde_cli_args(tmp_path, seed=None))
    with pytest.raises(ValueError, match="bound continuous-SDE process"):
        workflows.run_sde_simulation(_sde_cli_args(tmp_path, process="custom"))


def test_cli_sde_config_section_supplies_fields_but_never_parameters(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "sde_config.json"
    config_path.write_text(
        json.dumps({"sde": {"process": BOUND_SDE_PROCESS, "samples": 2, "steps": 3,
                            "dt_seconds": 1800.0, "initial_value": 0.5, "seed": 5}}),
        encoding="utf-8",
    )
    args = _sde_cli_args(
        tmp_path, config=str(config_path), process=None, samples=None, steps=None,
        dt=None, initial_value=None, seed=None,
    )

    workflows.run_sde_simulation(args)

    with open(tmp_path / "sde.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    trajectory = torch.tensor(payload[f"trajectory_{BOUND_SDE_STATE}"], dtype=torch.float64)
    assert trajectory.shape == (2, 4)
    assert payload["seed"] == 5 and payload["dt_seconds"] == 1800.0
    # Drift and diffusion coefficients are owned by the ParameterManager
    # database; a per-run override section is rejected before any output.
    config_path.write_text(
        json.dumps({"sde": {"parameters": {"volatility": 0.1}}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        workflows,
        "save_results",
        lambda *args, **kwargs: pytest.fail("rejected request wrote output"),
    )
    with pytest.raises(ValueError, match="ParameterManager database"):
        workflows.run_sde_simulation(_sde_cli_args(tmp_path, config=str(config_path)))


def test_execute_bound_continuous_sde_reports_exact_ar1_moments() -> None:
    result = workflows.execute_bound_continuous_sde(
        BOUND_SDE_PROCESS,
        initial_value=2.0,
        n_realizations=2,
        dt_seconds=6.0 * 3600.0,
        n_steps=3,
        seed=11,
    )
    alpha = result["parameters"]["reversion_rate"]
    sigma = result["parameters"]["volatility"]
    dt_days = 0.25
    phi = 1.0 - alpha * dt_days
    expected_mean = torch.tensor([2.0 * phi**n for n in range(4)], dtype=torch.float64)
    expected_variance = torch.tensor(
        [sigma * sigma * dt_days * (1.0 - phi ** (2 * n)) / (1.0 - phi * phi) for n in range(4)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        result[f"analytic_mean_{BOUND_SDE_STATE}"], expected_mean, rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(
        result[f"analytic_variance_{BOUND_SDE_STATE}"],
        expected_variance,
        rtol=1e-12,
        atol=0.0,
    )
    trajectory = result[f"trajectory_{BOUND_SDE_STATE}"]
    assert trajectory.shape == (2, 4)
    torch.testing.assert_close(trajectory[:, 0], torch.full((2,), 2.0, dtype=torch.float64))
    assert result["execution"]["step_count"] == 3
    assert result["state_units"] == {BOUND_SDE_STATE: "K"}
    assert result["initial_values"] == {BOUND_SDE_STATE: 2.0}


def test_execute_bound_marked_jump_reports_compound_poisson_moments_and_replays_the_functor() -> None:
    result = workflows.execute_bound_marked_jump(
        BOUND_JUMP_PROCESS,
        n_realizations=6,
        dt_seconds=6.0 * 3600.0,
        n_steps=4,
        seed=17,
    )
    assert result["semantic"] == "marked_jump"
    assert result["state_variables"] == [RAIN_COUNT, RAIN_DEPTH]
    assert result["state_units"] == {RAIN_COUNT: "1", RAIN_DEPTH: "mm"}
    rate = result["parameters"]["event_rate"]
    mean_depth = result["parameters"]["mean_depth"]
    index = torch.arange(5, dtype=torch.float64)
    expected_arrivals = rate * 0.25 * index
    torch.testing.assert_close(result[f"analytic_mean_{RAIN_COUNT}"], expected_arrivals, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(result[f"analytic_variance_{RAIN_COUNT}"], expected_arrivals, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(result[f"analytic_mean_{RAIN_DEPTH}"], mean_depth * expected_arrivals, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(
        result[f"analytic_variance_{RAIN_DEPTH}"], 2.0 * mean_depth**2 * expected_arrivals, rtol=1e-12, atol=0.0
    )
    reference = _reference_rainfall_trajectories(
        seed=17, n_realizations=6, dt_seconds=6.0 * 3600.0, n_steps=4, device=result["execution"]["device"]
    )
    for name in (RAIN_COUNT, RAIN_DEPTH):
        trajectory = result[f"trajectory_{name}"]
        assert trajectory.shape == (6, 5)
        torch.testing.assert_close(trajectory, reference[name], rtol=0.0, atol=0.0)
        assert len(result[f"ensemble_variance_{name}"]) == 5
    assert result["assertion_audit"]["structural_passed"] is True
    with pytest.raises(ValueError, match="bound marked-jump process"):
        workflows.execute_bound_marked_jump(BOUND_SDE_PROCESS, n_realizations=1, dt_seconds=3600.0, n_steps=1, seed=1)
    with pytest.raises(ValueError, match="bound continuous-SDE process"):
        workflows.execute_bound_continuous_sde(
            BOUND_JUMP_PROCESS, initial_value=0.0, n_realizations=1, dt_seconds=3600.0, n_steps=1, seed=1
        )
    with pytest.raises(ValueError, match="not bound"):
        workflows.execute_bound_stochastic_process(
            BOUND_JUMP_PROCESS,
            initial_values={"heat_storage": 1.0},
            n_realizations=1,
            dt_seconds=3600.0,
            n_steps=1,
            seed=1,
        )


def _gillespie_cli_args(tmp_path, **overrides) -> SimpleNamespace:
    values = {
        "config": None,
        "process": BOUND_JUMP_PROCESS,
        "samples": 3,
        "steps": 4,
        "dt": 6.0 * 3600.0,
        "seed": 5,
        "output": str(tmp_path / "gillespie"),
        "format": "json",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_gillespie_workflow_executes_the_bound_marked_jump_graph(tmp_path) -> None:
    workflows.run_gillespie_simulation(_gillespie_cli_args(tmp_path))

    with open(tmp_path / "gillespie.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    counts = torch.tensor(payload[f"trajectory_{RAIN_COUNT}"], dtype=torch.float64)
    depths = torch.tensor(payload[f"trajectory_{RAIN_DEPTH}"], dtype=torch.float64)
    assert counts.shape == (3, 5) and depths.shape == (3, 5)
    assert torch.equal(counts, torch.round(counts))
    assert bool((counts[:, 1:] >= counts[:, :-1]).all())
    assert payload["semantic"] == "marked_jump"
    assert payload["state_units"] == {RAIN_COUNT: "1", RAIN_DEPTH: "mm"}
    reference = _reference_rainfall_trajectories(
        seed=5, n_realizations=3, dt_seconds=6.0 * 3600.0, n_steps=4, device=payload["execution"]["device"]
    )
    torch.testing.assert_close(counts, reference[RAIN_COUNT], rtol=0.0, atol=0.0)
    torch.testing.assert_close(depths, reference[RAIN_DEPTH], rtol=0.0, atol=0.0)


def test_cli_gillespie_workflow_rejects_unseeded_unbound_and_reaction_declarations(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        workflows,
        "save_results",
        lambda *args, **kwargs: pytest.fail("rejected request wrote output"),
    )
    with pytest.raises(ValueError, match="seed"):
        workflows.run_gillespie_simulation(_gillespie_cli_args(tmp_path, seed=None))
    with pytest.raises(ValueError, match="bound marked-jump process"):
        workflows.run_gillespie_simulation(_gillespie_cli_args(tmp_path, process="custom"))
    config_path = tmp_path / "reactions.json"
    config_path.write_text(json.dumps({"gillespie": {"reactions": [{"name": "birth"}]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="arbitrary reaction networks"):
        workflows.run_gillespie_simulation(_gillespie_cli_args(tmp_path, config=str(config_path)))


EXAMPLE_STOCHASTIC_WEATHER_CONFIG = (
    Path(__file__).resolve().parents[3] / "examples" / "stochastic_weather_config.yaml"
)


@pytest.mark.parametrize(
    "command, section, process",
    [
        ("sde", "sde", BOUND_SDE_PROCESS),
        ("gillespie", "gillespie", BOUND_JUMP_PROCESS),
    ],
)
def test_example_stochastic_weather_config_drives_the_bound_processes(
    tmp_path, command, section, process
) -> None:
    """Every key of the example is read by the bound sde / gillespie command it names."""
    example = yaml.safe_load(EXAMPLE_STOCHASTIC_WEATHER_CONFIG.read_text(encoding="utf-8-sig"))
    assert set(example) == {"sde", "gillespie"}
    fields = example[section]
    args = SimpleNamespace(
        config=str(EXAMPLE_STOCHASTIC_WEATHER_CONFIG),
        process=None,
        samples=None,
        steps=None,
        dt=None,
        seed=None,
        output=str(tmp_path / command),
        format="json",
    )
    if command == "sde":
        args.initial_value = None
        workflows.run_sde_simulation(args)
    else:
        workflows.run_gillespie_simulation(args)

    with open(tmp_path / f"{command}.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["process"] == process == fields["process"]
    assert payload["seed"] == fields["seed"]
    assert payload["n_realizations"] == fields["samples"]
    assert payload["n_steps"] == fields["steps"]
    assert payload["dt_seconds"] == fields["dt_seconds"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_realizations": 0},
        {"n_steps": 0},
        {"dt_seconds": 0.0},
        {"initial_value": float("nan")},
        {"seed": -1},
        {"seed": 1.5},
    ],
)
def test_execute_bound_continuous_sde_rejects_invalid_requests(overrides) -> None:
    request = {
        "initial_value": 1.0,
        "n_realizations": 2,
        "dt_seconds": 3600.0,
        "n_steps": 2,
        "seed": 3,
    }
    request.update(overrides)
    with pytest.raises(ValueError):
        workflows.execute_bound_continuous_sde(BOUND_SDE_PROCESS, **request)


def test_api_sde_executes_the_bound_graph_and_reproduces_the_functor_path() -> None:
    response = asyncio.run(
        solve_sde(
            SDERequest(
                process=BOUND_SDE_PROCESS,
                initial_value=1.0,
                dt_seconds=3600.0,
                n_steps=3,
                n_realizations=2,
                seed=7,
            )
        )
    )

    assert response.status == "success"
    assert response.data is not None
    data = response.data
    assert data["process"] == BOUND_SDE_PROCESS
    assert data["semantic"] == "continuous_sde"
    assert data["state_variables"] == [BOUND_SDE_STATE]
    assert data["state_units"] == {BOUND_SDE_STATE: "K"}
    trajectory = torch.tensor(data["trajectories"][BOUND_SDE_STATE], dtype=torch.float64)
    assert trajectory.shape == (2, 4)
    reference = _reference_ou_trajectory(
        seed=7,
        initial_value=1.0,
        n_realizations=2,
        dt_seconds=3600.0,
        n_steps=3,
        device=data["execution"]["device"],
    )
    torch.testing.assert_close(trajectory, reference, rtol=0.0, atol=0.0)
    assert data["assertion_audit"]["structural_passed"] is True
    assert set(data["statistics"]) == {"time_seconds", "states"}
    statistics = data["statistics"]["states"][BOUND_SDE_STATE]
    assert set(statistics) == {
        "ensemble_mean",
        "ensemble_variance",
        "analytic_mean",
        "analytic_variance",
    }
    assert len(statistics["ensemble_mean"]) == 4


def test_api_sde_rejects_unbound_process_with_unprocessable_entity() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            solve_sde(
                SDERequest(
                    process="decomposition",
                    initial_value=1.0,
                    dt_seconds=3600.0,
                    n_steps=1,
                    n_realizations=1,
                    seed=7,
                )
            )
        )
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["type"] == "SDEValidationError"
    assert "bound continuous-SDE process" in exc_info.value.detail["message"]


def test_api_weather_fails_closed_without_graph_functor_binding(monkeypatch) -> None:
    monkeypatch.setattr(
        stochastic_api,
        "WeatherGenerator",
        lambda *args, **kwargs: pytest.fail(
            "fail-closed API constructed a standalone weather generator"
        ),
        raising=False,
    )

    response = asyncio.run(
        generate_weather(
            WeatherGenerationRequest(
                duration_days=2,
                n_realizations=2,
                variables=["temperature", "precipitation"],
                seed=7,
            )
        )
    )

    assert response.status == "error"
    assert "GraphIR/Functor runtime binding" in response.message
    assert response.data is not None
    error = response.data["error"]
    assert error.type == "WeatherGenerationValidationError"
    assert error.details is not None
    assert error.details["status"] == "[N]"
    assert {
        "declared weather-forcing operator",
        "StateManager forcing/state binding",
        "ParameterManager and unit binding",
        "simulator calendar and forcing-cursor binding",
        "durable simulator checkpoint binding",
        "restartable stochastic substream binding",
    } <= set(error.details["required_evidence"])


def test_api_stochastic_info_lists_only_real_routes() -> None:
    response = asyncio.run(stochastic_info())

    assert response.data is not None
    assert set(response.data["endpoints"]) == {"weather", "sde", "gillespie"}
    assert response.data["sde"]["status"] == "bound"
    assert response.data["sde"]["runtime_ready"] is True
    assert response.data["sde"]["bound_processes"] == [BOUND_SDE_PROCESS]
    assert response.data["supported_sde_processes"] == [BOUND_SDE_PROCESS]
    assert response.data["weather"]["status"] == "[N]"
    assert response.data["weather"]["runtime_ready"] is False
    assert response.data["gillespie"]["status"] == "bound"
    assert response.data["gillespie"]["runtime_ready"] is True
    assert response.data["gillespie"]["bound_processes"] == [BOUND_JUMP_PROCESS]
    assert response.data["supported_marked_jump_processes"] == [BOUND_JUMP_PROCESS]


def test_api_weather_fail_closed_contract_uses_http_not_implemented() -> None:
    app = FastAPI()
    app.include_router(stochastic_router, prefix="/stochastic")
    client = TestClient(app)

    response = client.post(
        "/stochastic/weather",
        json={
            "duration_days": 2,
            "n_realizations": 2,
            "variables": ["temperature", "precipitation"],
            "seed": 7,
        },
    )

    assert response.status_code == 501
    payload = response.json()
    assert payload["status"] == "error"
    assert "GraphIR/Functor runtime binding" in payload["message"]
    error = payload["data"]["error"]
    assert error["type"] == "WeatherGenerationValidationError"
    assert error["details"]["status"] == "[N]"

    # Client-side runtime flags are not request fields: they are rejected, never dropped.
    rejected = client.post(
        "/stochastic/weather",
        json={
            "duration_days": 2,
            "n_realizations": 2,
            "variables": ["temperature", "precipitation"],
            "seed": 7,
            "strict": False,
            "runtime_ready": True,
        },
    )
    assert rejected.status_code == 422
    assert "runtime_ready" in rejected.text


def test_api_sde_http_contract_executes_bound_process_and_rejects_unbound() -> None:
    app = FastAPI()
    app.include_router(stochastic_router, prefix="/stochastic")
    client = TestClient(app)

    response = client.post(
        "/stochastic/sde",
        json={
            "process": BOUND_SDE_PROCESS,
            "initial_value": 1.0,
            "dt_seconds": 3600.0,
            "n_steps": 2,
            "n_realizations": 2,
            "seed": 7,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert set(payload["data"]["parameters"]) == {"reversion_rate", "volatility"}
    assert len(payload["data"]["trajectories"][BOUND_SDE_STATE]) == 2

    overridden = client.post(
        "/stochastic/sde",
        json={
            "process": BOUND_SDE_PROCESS,
            "initial_value": 1.0,
            "parameters": {"volatility": 1.5},
            "dt_seconds": 3600.0,
            "n_steps": 2,
            "n_realizations": 2,
            "seed": 7,
        },
    )
    assert overridden.status_code == 422

    rejected = client.post(
        "/stochastic/sde",
        json={
            "process": "custom",
            "initial_value": 1.0,
            "dt_seconds": 3600.0,
            "n_steps": 2,
            "n_realizations": 2,
            "seed": 7,
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["type"] == "SDEValidationError"

    unseeded = client.post(
        "/stochastic/sde",
        json={
            "process": BOUND_SDE_PROCESS,
            "initial_value": 1.0,
            "dt_seconds": 3600.0,
            "n_steps": 2,
            "n_realizations": 2,
        },
    )
    assert unseeded.status_code == 422


def test_api_gillespie_executes_the_bound_marked_jump_and_rejects_reaction_payloads() -> None:
    app = FastAPI()
    app.include_router(stochastic_router, prefix="/stochastic")
    client = TestClient(app)

    response = client.post(
        "/stochastic/gillespie",
        json={
            "process": BOUND_JUMP_PROCESS,
            "dt_seconds": 6.0 * 3600.0,
            "n_steps": 3,
            "n_realizations": 2,
            "seed": 7,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    data = payload["data"]
    assert data["semantic"] == "marked_jump"
    assert data["state_variables"] == [RAIN_COUNT, RAIN_DEPTH]
    assert data["state_units"] == {RAIN_COUNT: "1", RAIN_DEPTH: "mm"}
    assert data["initial_values"] == {RAIN_COUNT: 0.0, RAIN_DEPTH: 0.0}
    counts = torch.tensor(data["trajectories"][RAIN_COUNT], dtype=torch.float64)
    depths = torch.tensor(data["trajectories"][RAIN_DEPTH], dtype=torch.float64)
    reference = _reference_rainfall_trajectories(
        seed=7, n_realizations=2, dt_seconds=6.0 * 3600.0, n_steps=3, device=data["execution"]["device"]
    )
    torch.testing.assert_close(counts, reference[RAIN_COUNT], rtol=0.0, atol=0.0)
    torch.testing.assert_close(depths, reference[RAIN_DEPTH], rtol=0.0, atol=0.0)
    assert set(data["statistics"]["states"]) == {RAIN_COUNT, RAIN_DEPTH}
    assert len(data["statistics"]["states"][RAIN_DEPTH]["analytic_variance"]) == 4

    rejected = client.post(
        "/stochastic/gillespie",
        json={
            "species": ["S"],
            "reactions": [{"name": "birth", "reactants": {}, "products": {"S": 1}, "rate_constant": 1.0}],
            "initial_state": {"S": 9},
            "final_time": 1.0,
            "n_realizations": 1,
            "seed": 7,
        },
    )
    assert rejected.status_code == 422

    unbound = client.post(
        "/stochastic/gillespie",
        json={"process": BOUND_SDE_PROCESS, "dt_seconds": 3600.0, "n_steps": 1, "n_realizations": 1, "seed": 7},
    )
    assert unbound.status_code == 422
    assert unbound.json()["detail"]["type"] == "GillespieValidationError"
    assert "bound marked-jump process" in unbound.json()["detail"]["message"]


def _args(results_path, observations_path, output_path, **overrides):
    values = {
        "results": str(results_path),
        "observations": str(observations_path),
        "output": str(output_path),
        "format": "json",
        "nominal_coverage": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validation_workflow_uses_torch_scalar_finiteness() -> None:
    assert not hasattr(workflows, "math")


def test_validation_workflow_restricts_tensor_payload_loading(
    monkeypatch,
    tmp_path,
) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    results_path.write_bytes(b"restricted-loader-test")
    observations_path.write_bytes(b"restricted-loader-test")
    payloads = iter(
        [
            {
                "air_temperature": torch.tensor(
                    [[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
                )
            },
            {"air_temperature": torch.tensor([0.0, 1.0])},
        ]
    )
    load_calls = []

    def restricted_load(path, *, map_location, weights_only):
        load_calls.append(
            {
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
            }
        )
        return next(payloads)

    monkeypatch.setattr(workflows.torch, "load", restricted_load)
    monkeypatch.setattr(workflows, "save_results", lambda *args: None)

    workflows.run_validation_analysis(
        _args(results_path, observations_path, tmp_path / "validation")
    )

    assert load_calls == [
        {
            "path": results_path,
            "map_location": "cpu",
            "weights_only": True,
        },
        {
            "path": observations_path,
            "map_location": "cpu",
            "weights_only": True,
        },
    ]


def test_validation_workflow_normalizes_ensemble_axis_and_saves_scalars(
    monkeypatch,
    tmp_path,
) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    torch.save(
        {
            "air_temperature": torch.tensor(
                [[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            )
        },
        results_path,
    )
    torch.save(
        {"air_temperature": torch.tensor([0.0, 1.0, 2.0])},
        observations_path,
    )
    captured = {}

    def capture_results(results, output_path, output_format):
        captured["results"] = results
        captured["output_path"] = output_path
        captured["format"] = output_format

    monkeypatch.setattr(workflows, "save_results", capture_results)

    workflows.run_validation_analysis(
        _args(results_path, observations_path, tmp_path / "validation")
    )

    summary = captured["results"]["air_temperature"]
    assert summary["pit_mean"] == pytest.approx(0.5)
    assert summary["crps_mean"] == pytest.approx(0.5)
    assert summary["coverage_probability"] == pytest.approx(1.0)
    assert summary["nominal_coverage"] == pytest.approx(0.5)
    assert summary["sample_size"] == 3
    assert all(
        isinstance(value, (int, float))
        for value in summary.values()
    )
    assert captured["format"] == "json"


def test_validation_workflow_accepts_observation_first_ensemble(
    monkeypatch,
    tmp_path,
) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    torch.save(
        {
            "air_temperature": torch.tensor(
                [[-1.0, 1.0], [0.0, 2.0], [1.0, 3.0]]
            )
        },
        results_path,
    )
    torch.save(
        {"air_temperature": torch.tensor([0.0, 1.0, 2.0])},
        observations_path,
    )
    captured = {}
    monkeypatch.setattr(
        workflows,
        "save_results",
        lambda results, *_: captured.update(results),
    )

    workflows.run_validation_analysis(
        _args(results_path, observations_path, tmp_path / "validation")
    )

    assert captured["air_temperature"]["crps_mean"] == pytest.approx(0.5)


def test_validation_workflow_rejects_ambiguous_square_tensor(
    monkeypatch,
    tmp_path,
) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    torch.save(
        {"air_temperature": torch.arange(9, dtype=torch.float32).reshape(3, 3)},
        results_path,
    )
    torch.save(
        {"air_temperature": torch.tensor([0.0, 1.0, 2.0])},
        observations_path,
    )
    monkeypatch.setattr(workflows, "save_results", lambda *args: None)

    with pytest.raises(ValueError, match="ambiguous"):
        workflows.run_validation_analysis(
            _args(results_path, observations_path, tmp_path / "validation")
        )


def test_cli_validation_uses_netcdf_sample_and_time_dimensions(
    monkeypatch,
    tmp_path,
) -> None:
    results_path = tmp_path / "results.nc"
    observations_path = tmp_path / "observations.nc"
    xr.Dataset(
        {
            "air_temperature": (
                ("sample", "time"),
                [
                    [-2.0, 9.0, 99.0],
                    [0.0, 11.0, 101.0],
                    [2.0, 13.0, 103.0],
                ],
            )
        }
    ).to_netcdf(results_path)
    xr.Dataset(
        {"air_temperature": (("time",), [0.0, 11.0, 101.0])}
    ).to_netcdf(observations_path)
    captured = {}
    monkeypatch.setattr(
        workflows,
        "save_results",
        lambda results, *_: captured.update(results),
    )
    args = _args(
        results_path,
        observations_path,
        tmp_path / "validation",
    )
    args.stochastic_command = "validate"

    assert execute(args, {}) is True

    summary = captured["air_temperature"]
    assert summary["crps_mean"] == pytest.approx(4.0 / 9.0)
    assert summary["coverage_probability"] == pytest.approx(1.0)
    assert summary["sample_size"] == 3


def test_validation_workflow_writes_json_summary(tmp_path) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    output_path = tmp_path / "validation"
    torch.save(
        {
            "air_temperature": torch.tensor(
                [[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
            )
        },
        results_path,
    )
    torch.save(
        {"air_temperature": torch.tensor([0.0, 1.0, 2.0])},
        observations_path,
    )

    workflows.run_validation_analysis(
        _args(results_path, observations_path, output_path)
    )

    with output_path.with_suffix(".json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["air_temperature"]["sample_size"] == 3
    assert saved["air_temperature"]["coverage_probability"] == pytest.approx(
        1.0
    )


def test_validation_workflow_rejects_incompatible_ensemble_shape(
    monkeypatch,
    tmp_path,
) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    torch.save(
        {"air_temperature": torch.ones(2, 4)},
        results_path,
    )
    torch.save(
        {"air_temperature": torch.ones(3)},
        observations_path,
    )
    monkeypatch.setattr(workflows, "save_results", lambda *args: None)

    with pytest.raises(ValueError, match="observation count"):
        workflows.run_validation_analysis(
            _args(results_path, observations_path, tmp_path / "validation")
        )


@pytest.mark.parametrize(
    "nominal",
    [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), -float("inf")],
)
def test_validation_workflow_rejects_invalid_nominal_coverage(
    monkeypatch,
    tmp_path,
    nominal: float,
) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    torch.save(
        {"air_temperature": torch.ones(2, 3)},
        results_path,
    )
    torch.save(
        {"air_temperature": torch.ones(3)},
        observations_path,
    )
    monkeypatch.setattr(workflows, "save_results", lambda *args: None)

    with pytest.raises(ValueError, match="nominal_coverage"):
        workflows.run_validation_analysis(
            _args(
                results_path,
                observations_path,
                tmp_path / "validation",
                nominal_coverage=nominal,
            )
        )


def test_uncertainty_config_declares_the_correlation_nugget_explicitly() -> None:
    declared = workflows._parse_distribution_spec(
        "p",
        {
            "distribution": "normal",
            "params": {"loc": 1.0, "scale": 0.5},
            "correlation_length": 10.0,
            "kernel": "matern_3_2",
            "correlation_nugget": 1.0e-6,
        },
    )
    uncorrelated = workflows._parse_distribution_spec(
        "p",
        {"distribution": "normal", "params": {"loc": 1.0, "scale": 0.5}},
    )

    assert declared.correlation_nugget == pytest.approx(1.0e-6)
    assert declared.kernel.value == "matern_3_2"
    assert uncorrelated.correlation_nugget is None
    assert uncorrelated.kernel is None


@pytest.mark.parametrize(
    "missing, message", [("kernel", "kernel"), ("correlation_nugget", "correlation_nugget")]
)
def test_uncertainty_config_of_a_correlated_field_states_kernel_and_nugget(
    missing: str, message: str
) -> None:
    entry = {
        "distribution": "normal",
        "params": {"loc": 1.0, "scale": 0.5},
        "correlation_length": 10.0,
        "kernel": "exponential",
        "correlation_nugget": 0.0,
    }
    entry.pop(missing)
    with pytest.raises(ValueError, match=message):
        workflows._parse_distribution_spec("p", entry)


def test_uncertainty_config_rejects_unknown_parameter_keys() -> None:
    """A misspelled key (here the correlation length) is an error, not ignored."""
    with pytest.raises(ValueError, match="corelation_length"):
        workflows._build_parameter_specs(
            {
                "p": {
                    "distribution": "normal",
                    "params": {"loc": 1.0, "scale": 0.5},
                    "corelation_length": 10.0,
                }
            }
        )


def test_validation_without_observations_is_an_error_not_an_empty_result(
    monkeypatch,
    tmp_path,
) -> None:
    """PIT, CRPS and coverage compare the ensemble with observations."""
    results_path = tmp_path / "results.pt"
    torch.save({"air_temperature": torch.ones(4, 3)}, results_path)
    saved = []
    monkeypatch.setattr(workflows, "save_results", lambda *args: saved.append(args))

    with pytest.raises(ValueError, match="observations"):
        workflows.run_validation_analysis(
            _args(results_path, None, tmp_path / "validation", observations=None)
        )
    assert saved == []


def test_validation_without_a_matching_variable_is_an_error(monkeypatch, tmp_path) -> None:
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    torch.save({"air_temperature": torch.ones(4, 3)}, results_path)
    torch.save({"soil_moisture": torch.ones(3)}, observations_path)
    saved = []
    monkeypatch.setattr(workflows, "save_results", lambda *args: saved.append(args))

    with pytest.raises(KeyError, match="air_temperature"):
        workflows.run_validation_analysis(
            _args(results_path, observations_path, tmp_path / "validation")
        )
    assert saved == []


@pytest.mark.parametrize(
    "suffix, text",
    [(".json", "[1, 2]"), (".json", "3"), (".yaml", "- 1\n- 2\n"), (".yaml", "3\n")],
)
def test_stochastic_config_that_is_not_a_mapping_is_rejected(tmp_path, suffix, text) -> None:
    from relic.stochastic.config import load_config

    path = tmp_path / f"stochastic{suffix}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(str(path))


def test_stochastic_config_section_that_is_not_a_mapping_is_rejected() -> None:
    from relic.stochastic.config import get_section

    assert get_section({}, "sde") == {}
    with pytest.raises(TypeError, match="sde"):
        get_section({"sde": ["n_steps", 10]}, "sde")


def test_bound_process_needs_an_initial_value_for_every_state() -> None:
    """A missing initial value is an error, not an implicit zero state."""
    with pytest.raises(ValueError, match="initial_values.*missing"):
        workflows.execute_bound_stochastic_process(
            BOUND_SDE_PROCESS,
            initial_values={},
            n_realizations=1,
            dt_seconds=3600.0,
            n_steps=1,
            seed=1,
        )


def test_uncertainty_sampling_uses_the_configured_seed(monkeypatch, tmp_path) -> None:
    """uncertainty.seed from the config file is honoured like sde.seed and gillespie.seed."""
    config_path = tmp_path / "stochastic.json"
    config_path.write_text(
        json.dumps(
            {
                "uncertainty": {
                    "samples": 4,
                    "seed": 17,
                    "parameters": {
                        "p": {"distribution": "normal", "params": {"loc": 1.0, "scale": 0.5}}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    saved = []
    monkeypatch.setattr(workflows, "save_results", lambda results, *args: saved.append(results))
    args = SimpleNamespace(
        config=str(config_path), samples=None, seed=None, output=str(tmp_path / "u"), format="json"
    )

    workflows.run_uncertainty_analysis(args)
    workflows.run_uncertainty_analysis(args)

    assert torch.equal(saved[0]["p"], saved[1]["p"])


def _uncertainty_args(tmp_path, parameters):
    config_path = tmp_path / "stochastic.json"
    config_path.write_text(
        json.dumps({"uncertainty": {"samples": 4, "seed": 3, "parameters": parameters}}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        config=str(config_path), samples=None, seed=None, output=str(tmp_path / "u"), format="netcdf"
    )


def test_uncertainty_netcdf_uses_the_declared_parameter_units(tmp_path) -> None:
    parameters = {
        "p": {"distribution": "normal", "params": {"loc": 1.0, "scale": 0.5}, "units": "kg m-2"}
    }
    workflows.run_uncertainty_analysis(_uncertainty_args(tmp_path, parameters))

    with xr.open_dataset(tmp_path / "u.nc") as dataset:
        assert dataset["p"].attrs["units"] == "kg m-2"
        assert dataset["p"].dims == ("sample", "location")


def test_uncertainty_netcdf_without_declared_units_raises(tmp_path) -> None:
    parameters = {"p": {"distribution": "normal", "params": {"loc": 1.0, "scale": 0.5}}}
    with pytest.raises(ValueError, match="'p' has no declared units"):
        workflows.run_uncertainty_analysis(_uncertainty_args(tmp_path, parameters))


# ---------------------------------------------------------------------------
# Round 5: the nominal coverage is the stated value or, when None, the
# parameter-database value validation.stochastic_scores.nominal_interval_coverage;
# a namespace without the attribute is an error (no getattr default).
# ---------------------------------------------------------------------------


def _coverage_files(tmp_path):
    results_path = tmp_path / "results.pt"
    observations_path = tmp_path / "observations.pt"
    torch.save({"air_temperature": torch.tensor([[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]])}, results_path)
    torch.save({"air_temperature": torch.tensor([0.0, 1.0, 2.0])}, observations_path)
    return results_path, observations_path


def test_validation_nominal_coverage_none_reads_the_parameter_database(monkeypatch, tmp_path) -> None:
    from relic.parameters import get_global_manager

    results_path, observations_path = _coverage_files(tmp_path)
    captured = {}
    monkeypatch.setattr(workflows, "save_results", lambda results, *_: captured.update(results))
    manager = get_global_manager()
    key = "validation.stochastic_scores.nominal_interval_coverage"
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, 0.8)
        workflows.run_validation_analysis(
            _args(results_path, observations_path, tmp_path / "validation", nominal_coverage=None)
        )
    finally:
        manager.set_value(key, original)
    assert captured["air_temperature"]["nominal_coverage"] == pytest.approx(0.8)


def test_validation_namespace_without_nominal_coverage_is_an_error(monkeypatch, tmp_path) -> None:
    results_path, observations_path = _coverage_files(tmp_path)
    monkeypatch.setattr(workflows, "save_results", lambda *args: None)
    args = _args(results_path, observations_path, tmp_path / "validation")
    del args.nominal_coverage
    with pytest.raises(AttributeError, match="nominal_coverage"):
        workflows.run_validation_analysis(args)
