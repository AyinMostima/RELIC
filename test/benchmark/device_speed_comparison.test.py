"""Device speed comparison benchmark with real RELIC scientific functors."""

from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.scientific.atmosphere.environmental_forcing import (
    EnvironmentalForcingFunctor,
)
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)


pytestmark = [pytest.mark.perf]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_tensor_grid(
    value: float,
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.full(shape, value, dtype=dtype, device=device)


def _build_initial_state(
    device: torch.device,
    grid_shape: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    rows, cols = grid_shape
    base = torch.linspace(
        0.0,
        1.0,
        steps=rows * cols,
        device=device,
        dtype=torch.float32,
    ).reshape(rows, cols)

    air_temperature = 286.15 + 12.0 * base
    surface_temperature = 288.15 + 10.0 * base
    relative_humidity = torch.clamp(0.35 + 0.55 * (1.0 - base), min=0.05, max=0.98)
    wind_speed = 1.0 + 4.0 * base
    incoming_shortwave_potential = 150.0 + 450.0 * base
    cos_zenith = torch.clamp(0.1 + 0.6 * base, min=0.05, max=0.85)

    return {
        "air_temperature": air_temperature,
        "surface_temperature": surface_temperature,
        "relative_humidity": relative_humidity,
        "wind_speed": wind_speed,
        "atmospheric_pressure": _to_tensor_grid(101325.0, grid_shape, device),
        "incoming_shortwave_potential": incoming_shortwave_potential,
        "cos_zenith": cos_zenith,
        "cloud_fraction": _to_tensor_grid(0.35, grid_shape, device),
        "surface_albedo": _to_tensor_grid(0.2, grid_shape, device),
        "surface_emissivity": _to_tensor_grid(0.96, grid_shape, device),
        "atmospheric_co2_ppm": _to_tensor_grid(415.0, grid_shape, device),
        "leaf_area_index": _to_tensor_grid(3.0, grid_shape, device),
        "vcmax_scaling": _to_tensor_grid(1.0, grid_shape, device),
        "jmax_scaling": _to_tensor_grid(1.0, grid_shape, device),
        "water_stress_factor": _to_tensor_grid(1.0, grid_shape, device),
        "temperature_stress_factor": _to_tensor_grid(1.0, grid_shape, device),
        "ozone_stress_factor": _to_tensor_grid(1.0, grid_shape, device),
        "surface_resistance": _to_tensor_grid(120.0, grid_shape, device),
    }


def _force_functor_device(functor: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    def _move_value(value: object) -> object:
        if isinstance(value, torch.nn.Parameter):
            return value
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _move_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_move_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_move_value(v) for v in value)
        return value

    functor.to(device)
    if hasattr(functor, "device"):
        setattr(functor, "device", device)
    for module in functor.modules():
        for attr_name, attr_value in list(vars(module).items()):
            moved = _move_value(attr_value)
            if moved is attr_value:
                continue
            try:
                setattr(module, attr_name, moved)
            except (AttributeError, TypeError):
                continue
        if hasattr(module, "device"):
            setattr(module, "device", device)
    return functor


def _assert_registered_functor_devices(
    simulator: RELICSimulator,
    expected_device: torch.device,
) -> None:
    for functor_name, functor in simulator.functors.items():
        actual_device = getattr(functor, "device", None)
        if actual_device is None:
            raise AssertionError(f"Functor {functor_name} has no device attribute")
        resolved = torch.device(actual_device)
        if resolved.type != expected_device.type:
            raise AssertionError(
                f"Functor {functor_name} device mismatch: {resolved} != {expected_device}"
            )


def _build_simulator(device: torch.device, grid_shape: Tuple[int, int]) -> RELICSimulator:
    simulator = RELICSimulator(
        {
            "time_step": 60.0,
            "conservation_checks": False,
            "strict_conservation_enabled": False,
            "strict_dependencies": False,
            "enable_bootstrap_modules": False,
            "device": str(device),
            "strict_scale_detection": False,
            "enable_auto_scale": False,
        }
    )
    simulator.initialize_state(_build_initial_state(device=device, grid_shape=grid_shape))

    environmental_forcing = _force_functor_device(
        EnvironmentalForcingFunctor(
            name="environmental_forcing",
            device=device,
        ),
        device=device,
    )
    radiation = _force_functor_device(
        RadiationFunctor(
            name="atmospheric_radiation",
            config={"enable_conservation_check": False},
            device=device,
        ),
        device=device,
    )
    penman = _force_functor_device(
        PenmanMonteithET(name="penman_monteith_et", config={"canopy_resistance_source": "state"}),
        device=device,
    )
    farquhar = _force_functor_device(
        FarquharPhotosynthesis(
            name="farquhar_photosynthesis",
            device=device,
            enable_gpu=(device.type == "cuda"),
        ),
        device=device,
    )
    simulator.register_functor(environmental_forcing, execution_group="fast")
    simulator.register_functor(radiation, execution_group="fast")
    simulator.register_functor(penman, execution_group="fast")
    simulator.register_functor(farquhar, execution_group="fast")
    _assert_registered_functor_devices(simulator, expected_device=device)
    return simulator


def _assert_canonical_chain_outputs(
    simulator: RELICSimulator,
    device: torch.device,
    grid_shape: Tuple[int, int],
) -> None:
    """Verify canonical derived ports, devices, and writer provenance."""
    expected_writers = {
        "vapor_pressure_deficit": "environmental_forcing",
        "ppfd": "atmospheric_radiation",
        "evapotranspiration_flux": "penman_monteith_et",
        "latent_heat_flux": "penman_monteith_et",
        "gross_primary_productivity": "farquhar_photosynthesis",
    }
    state = simulator.state_manager.get_state()
    for variable_name, writer_name in expected_writers.items():
        value = state[variable_name]
        assert value.shape == grid_shape
        assert value.device.type == device.type
        assert torch.isfinite(value).all()
        provenance = simulator.get_variable_provenance(variable_name)
        assert provenance
        assert {record["functor"] for record in provenance} == {writer_name}


def _benchmark_device(
    device: torch.device,
    grid_shape: Tuple[int, int],
    warmup_steps: int,
    timed_steps: int,
    repeats: int,
) -> Dict[str, float]:
    step_timings: List[float] = []
    initialization_timings: List[float] = []
    warmup_timings: List[float] = []
    for _ in range(repeats):
        init_start = time.perf_counter()
        simulator = _build_simulator(device=device, grid_shape=grid_shape)
        _sync_if_cuda(device)
        initialization_timings.append(time.perf_counter() - init_start)
        warmup_start = time.perf_counter()
        for _ in range(warmup_steps):
            simulator.step(dt=60.0)
        _sync_if_cuda(device)
        warmup_timings.append(time.perf_counter() - warmup_start)
        _assert_canonical_chain_outputs(simulator, device, grid_shape)
        for _ in range(timed_steps):
            _sync_if_cuda(device)
            start = time.perf_counter()
            simulator.step(dt=60.0)
            _sync_if_cuda(device)
            step_timings.append(time.perf_counter() - start)

    ordered = sorted(step_timings)
    p95_position = (len(ordered) - 1) * 0.95
    lower = int(p95_position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = p95_position - lower
    p95 = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    median_step = statistics.median(step_timings)
    return {
        "median_step_s": median_step,
        "p95_step_s": p95,
        "mean_step_s": statistics.mean(step_timings),
        "std_step_s": statistics.pstdev(step_timings),
        "median_initialization_s": statistics.median(initialization_timings),
        "median_warmup_s": statistics.median(warmup_timings),
        "steps_per_second": 1.0 / median_step if median_step > 0 else 0.0,
    }


def test_device_speed_comparison_with_real_functors() -> None:
    """Compare CPU and CUDA throughput using a real multi-functor scientific chain."""
    torch.manual_seed(42)
    grid_shape = (96, 96)
    warmup_steps = 3
    timed_steps = 10
    repeats = 3
    chain = [
        "EnvironmentalForcingFunctor",
        "RadiationFunctor",
        "PenmanMonteithET",
        "FarquharPhotosynthesis",
    ]

    print(
        "\n[Benchmark] Real functor chain: "
        + " -> ".join(chain)
        + f" | grid={grid_shape[0]}x{grid_shape[1]}"
    )

    cpu_device = torch.device("cpu")
    cpu_metrics = _benchmark_device(
        device=cpu_device,
        grid_shape=grid_shape,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        repeats=repeats,
    )
    print(
        "[Benchmark] CPU "
        f"median={cpu_metrics['median_step_s']:.6f}s/step "
        f"p95={cpu_metrics['p95_step_s']:.6f}s/step "
        f"mean={cpu_metrics['mean_step_s']:.6f}s/step "
        f"throughput={cpu_metrics['steps_per_second']:.2f} steps/s"
    )
    assert cpu_metrics["steps_per_second"] > 0.0

    if not torch.cuda.is_available():
        print("[Benchmark] CUDA not available; CPU baseline only.")
        return

    cuda_device = torch.device("cuda")
    cuda_metrics = _benchmark_device(
        device=cuda_device,
        grid_shape=grid_shape,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        repeats=repeats,
    )

    speedup = (
        cuda_metrics["steps_per_second"] / cpu_metrics["steps_per_second"]
        if cpu_metrics["steps_per_second"] > 0.0
        else 0.0
    )
    print(
        "[Benchmark] CUDA "
        f"median={cuda_metrics['median_step_s']:.6f}s/step "
        f"p95={cuda_metrics['p95_step_s']:.6f}s/step "
        f"mean={cuda_metrics['mean_step_s']:.6f}s/step "
        f"throughput={cuda_metrics['steps_per_second']:.2f} steps/s "
        f"speedup_vs_cpu={speedup:.2f}x"
    )
    assert cuda_metrics["steps_per_second"] > 0.0
