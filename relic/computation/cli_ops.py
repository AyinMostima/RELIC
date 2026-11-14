"""Compute CLI operations delegated from interface layer."""

from __future__ import annotations

import logging
from typing import Dict, Any

from relic.utils.imports import module_available

logger = logging.getLogger(__name__)


def _resolve_io(io: Dict[str, Any]):
    """Resolve required IO callbacks from the interface layer."""
    return (
        io["print_success"],
        io["print_error"],
        io["print_warning"],
        io["print_info"],
        io["print_header"],
        io["save_output"],
        io["ProgressReporter"],
        io["check_experimental_feature"],
    )


def execute(args, config: Dict[str, Any], io: Dict[str, Any]) -> bool:
    """Execute compute command with delegated IO handlers."""
    (
        print_success,
        print_error,
        print_warning,
        print_info,
        print_header,
        save_output,
        ProgressReporter,
        check_experimental_feature,
    ) = _resolve_io(io)
    if not hasattr(args, "compute_command") or not args.compute_command:
        print_error("Compute command required")
        return False

    command = args.compute_command

    if command == "health":
        return run_health_check(args, config, io)
    if command == "bench":
        return run_benchmark(args, config, io)
    if command == "ddp-check":
        return run_ddp_check(args, config, io)
    if command == "splitting-sensitivity":
        return run_splitting_sensitivity(args, config, io)
    if command == "dynamical-benchmark":
        return run_dynamical_benchmark(args, config, io)

    print_error(f"Unknown compute command: {command}")
    return False


def _print_dependency_status(name: str, available: bool, version: str = None, io: Dict[str, Any] = None):
    """Print dependency status using provided IO handlers."""
    print_success, print_error, _, _, _, _, _, _ = _resolve_io(io)
    if available:
        status_msg = f"  {name}: Available"
        if version:
            status_msg += f" ({version})"
        print_success(status_msg)
    else:
        print_error(f"  {name}: Not available")


def run_health_check(args, config: Dict[str, Any], io: Dict[str, Any]) -> bool:
    """Run computational health check."""
    (
        print_success,
        print_error,
        print_warning,
        print_info,
        print_header,
        save_output,
        ProgressReporter,
        check_experimental_feature,
    ) = _resolve_io(io)
    if not module_available("relic.computation") or not module_available("relic.computation.gpu"):
        print_error("Health check failed: required computation modules are unavailable")
        return False

    from relic.computation import healthcheck
    from relic.computation.gpu import get_memory_usage

    print_header("Computational Health Check")

    with ProgressReporter(description="Checking computation layer") as progress:
        status = healthcheck(
            verbose=args.verbose,
            runtime_probe=bool(args.verbose),
        )
        progress.update(1, "Health check complete")

    print(f"\nOverall Status: {status['overall'].upper()}")

    deps = status["dependencies"]
    print("\nCore Dependencies:")
    _print_dependency_status("PyTorch", deps.get("torch", False), deps.get("torch_version"), io=io)
    _print_dependency_status("CUDA", deps.get("cuda", False), deps.get("cuda_version"), io=io)
    compile_status = status.get("capabilities", {}).get("torch_compile", {})
    _print_dependency_status(
        "JIT Compilation",
        compile_status.get("runtime_ready") is True,
        io=io,
    )
    _print_dependency_status("Distributed", deps.get("distributed", False), io=io)
    _print_dependency_status("AMP", deps.get("amp", False), deps.get("amp"), io=io)
    _print_dependency_status("Parallel Processing", deps.get("parallel", False), io=io)

    gpu_query_failed = False
    if deps.get("cuda", False) and args.verbose:
        print("\nGPU Information:")
        devices = deps.get("cuda_devices", [])
        for i, device in enumerate(devices):
            print(f"  CUDA:{i} - {device['name']} ({device['memory']}GB)")
        try:
            memory = get_memory_usage()
            print("\nGPU Memory Status:")
            print(f"  Allocated: {memory['allocated']:.2f} GB")
            print(f"  Reserved:  {memory['reserved']:.2f} GB")
            print(f"  Free:      {memory['free']:.2f} GB")
        except RuntimeError as e:
            # CLI boundary: a failed CUDA memory query is reported as an
            # error and makes the health command fail.
            print_error(f"  GPU memory query failed: {e}")
            gpu_query_failed = True

    if status.get("warnings"):
        print("\nWarnings:")
        for warning in status["warnings"]:
            print_warning(f"  - {warning}")

    if status.get("recommendations"):
        print("\nRecommendations:")
        for rec in status["recommendations"]:
            print_info(f"  - {rec}")

    if args.output:
        save_output(status, args.output, args.format)

    return status["overall"] in ["healthy", "degraded"] and not gpu_query_failed


def run_benchmark(args, config: Dict[str, Any], io: Dict[str, Any]) -> bool:
    """Run performance benchmark."""
    (
        print_success,
        print_error,
        print_warning,
        print_info,
        print_header,
        save_output,
        ProgressReporter,
        check_experimental_feature,
    ) = _resolve_io(io)
    if not module_available("relic.computation") or not module_available("relic.computation.jit"):
        print_error("Benchmark failed: required computation modules are unavailable")
        return False

    import time
    import torch
    from relic.computation import healthcheck, select_device
    from relic.computation.jit import maybe_compile
    from relic.computation.benchmarks import (
        SYNTHETIC_BENCHMARK_SCOPE,
        create_benchmark_function,
        create_test_inputs,
    )

    print_header(f"Performance Benchmark - {args.target}")

    device = select_device(args.device)
    print_info(f"Running benchmark on device: {device}")

    computation_config = config.get("computation", {})
    if not isinstance(computation_config, dict):
        computation_config = {}
    compile_backend = str(computation_config.get("compile_backend", "inductor"))
    compile_mode = str(computation_config.get("compile_mode", "default"))

    if args.jit:
        compile_health = healthcheck(
            runtime_probe=True,
            compile_backend=compile_backend,
            compile_device=str(device),
        )
        compile_status = compile_health.get("capabilities", {}).get(
            "torch_compile",
            {},
        )
        if compile_status.get("runtime_ready") is not True:
            blocker = compile_status.get("blocker") or {}
            blocker_type = blocker.get("type", "RuntimeUnavailable")
            blocker_message = blocker.get("message", "compile probe did not pass")
            print_error(
                "JIT benchmark blocked: "
                f"{blocker_type}: {blocker_message}"
            )
            if args.output:
                save_output(
                    {
                        "status": "blocked",
                        "benchmark_scope": SYNTHETIC_BENCHMARK_SCOPE,
                        "scientific_equivalence": False,
                        "scientific_runtime_owner": None,
                        "target_label": args.target,
                        "capability": compile_status,
                    },
                    args.output,
                    "json",
                )
            return False

    benchmark_func = create_benchmark_function(args.target)
    inputs = create_test_inputs(args.target, args.batch_size, args.n_points, device)

    modes = ["off"]
    if args.jit:
        modes.extend(["compile"])

    results = {
        "benchmark_scope": getattr(
            benchmark_func,
            "relic_benchmark_scope",
            SYNTHETIC_BENCHMARK_SCOPE,
        ),
        "scientific_equivalence": False,
        "scientific_runtime_owner": None,
        "target_label": args.target,
        "environment": {
            "device": str(device),
            "compile_backend": compile_backend if args.jit else None,
            "compile_mode": compile_mode if args.jit else None,
        },
        "modes": {},
    }

    def _invoke_benchmark(func, benchmark_inputs):
        if isinstance(benchmark_inputs, dict):
            return func(**benchmark_inputs)
        if isinstance(benchmark_inputs, (list, tuple)):
            return func(*benchmark_inputs)
        return func(benchmark_inputs)

    with ProgressReporter(total=len(modes), description="Running benchmarks") as progress:
        for mode in modes:
            if mode == "compile":
                compiled_func = maybe_compile(
                    benchmark_func,
                    mode="compile",
                    backend=compile_backend,
                    compile_mode=compile_mode,
                )
            else:
                compiled_func = benchmark_func

            for _ in range(3):
                _ = _invoke_benchmark(compiled_func, inputs)

            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.perf_counter()
            for _ in range(10):
                _ = _invoke_benchmark(compiled_func, inputs)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.perf_counter() - start_time

            avg_time = elapsed / 10.0
            throughput = args.batch_size / avg_time

            results["modes"][mode] = {
                "avg_time_s": avg_time,
                "throughput_samples_per_s": throughput,
            }

            progress.update(1, f"Completed {mode} benchmark")

    print("\nBenchmark Results:")
    for mode, stats in results["modes"].items():
        print(f"  {mode}: {stats['avg_time_s']:.6f}s per run, {stats['throughput_samples_per_s']:.2f} samples/s")

    if args.output:
        save_output(results, args.output, "json")

    return True


def run_ddp_check(args, config: Dict[str, Any], io: Dict[str, Any]) -> bool:
    """Run distributed computing check."""
    (
        print_success,
        print_error,
        print_warning,
        print_info,
        print_header,
        save_output,
        ProgressReporter,
        check_experimental_feature,
    ) = _resolve_io(io)
    if not check_experimental_feature("Distributed Computing", "RELIC_EXPERIMENTAL_DISTRIBUTED"):
        print_warning("Distributed computing check requires RELIC_EXPERIMENTAL_DISTRIBUTED=1")
        return False

    import subprocess
    import sys

    print_header(f"Distributed Computing Check (world_size={args.world_size})")

    python_exe = sys.executable
    script_path = "-c"
    script_code = "from relic.computation import ddp_healthcheck; ddp_healthcheck()"

    procs = []
    with ProgressReporter(total=args.world_size, description="Launching processes") as progress:
        for rank in range(args.world_size):
            cmd = [
                python_exe,
                script_path,
                script_code,
                "--rank",
                str(rank),
                "--world-size",
                str(args.world_size),
            ]
            proc = subprocess.Popen(cmd)
            procs.append(proc)
            progress.update(1, f"Launched rank {rank}")

    success = True
    with ProgressReporter(total=args.world_size, description="Waiting for completion") as progress:
        for i, proc in enumerate(procs):
            proc.wait()
            if proc.returncode != 0:
                print_error(f"Rank {i} failed with return code {proc.returncode}")
                success = False
            else:
                print_success(f"Rank {i} completed successfully")
            progress.update(1, f"Rank {i} complete")

    if success:
        print_success("\nDistributed computing check PASSED")
    else:
        print_error("\nDistributed computing check FAILED")

    results = {
        "world_size": args.world_size,
        "success": success,
    }
    if args.output:
        save_output(results, args.output, "json")

    return success


def run_splitting_sensitivity(args, config: Dict[str, Any], io: Dict[str, Any]) -> bool:
    """Run splitting sensitivity diagnostic using an existing simulator instance."""
    (
        print_success,
        print_error,
        print_warning,
        print_info,
        print_header,
        save_output,
        ProgressReporter,
        check_experimental_feature,
    ) = _resolve_io(io)

    print_header("Splitting Sensitivity Diagnostic")
    simulator = config.get("simulator")
    if simulator is None:
        print_error("splitting-sensitivity requires config['simulator'] with a RELICSimulator instance")
        return False

    from relic.core.simulator.execution_loop import compute_splitting_sensitivity

    state = config.get("state")
    if state is None:
        state = simulator.get_state()

    # args is the namespace resolved by relic.interfaces.cli.compute: command
    # line values, or the computation.diagnostics.* parameter-database values
    # for omitted settings. No second default is kept here.
    provided_order = args.group_order
    if provided_order:
        group_order = [item.strip() for item in provided_order.split(",") if item.strip()]
    else:
        # The fast-group order of the committed provider view (public API);
        # an unpublished simulator has no order to scan.
        view = simulator.get_committed_provider_view()
        if view is None:
            print_error(
                "splitting-sensitivity needs --group-order or a simulator with a committed "
                "provider view (register the Functors and publish the view first)"
            )
            return False
        group_order = list(view.order_fast)

    if not group_order:
        print_error("No functor order available. Provide --group-order or initialize fast group order.")
        return False

    dt = float(args.dt)
    n_permutations = int(args.n_permutations)

    result = compute_splitting_sensitivity(
        simulator=simulator,
        state=state,
        dt=dt,
        group_order=group_order,
        n_permutations=n_permutations,
    )
    print_info(f"Permutations tested: {result['permutations_tested']}")
    print_info(f"Commutativity violation: {result['commutativity_violation']:.6e}")

    if args.output:
        save_output(result, args.output, "json")
    return True


def run_dynamical_benchmark(args, config: Dict[str, Any], io: Dict[str, Any]) -> bool:
    """Run dynamical feature-space benchmark on synthetic trajectories."""
    (
        print_success,
        print_error,
        print_warning,
        print_info,
        print_header,
        save_output,
        ProgressReporter,
        check_experimental_feature,
    ) = _resolve_io(io)

    if not module_available("relic.computation.benchmarks"):
        print_error("Dynamical benchmark failed: benchmark module is unavailable")
        return False

    import math
    import torch
    from relic.computation.benchmarks import (
        compute_dynamical_embedding,
        compute_mode_similarity_score,
        compute_spectral_signature,
        compute_structure_function,
    )

    print_header("Dynamical Feature Benchmark")
    # Settings resolved by relic.interfaces.cli.compute (command line or
    # computation.diagnostics.dynamical_benchmark.*); no second default here.
    series_length = int(args.series_length)
    dt = float(args.dt)
    dominant_frequency = float(args.frequency)

    t = torch.arange(series_length, dtype=torch.float32) * dt
    omega = 2.0 * math.pi * dominant_frequency
    signal = torch.sin(omega * t)
    auxiliary = torch.cos(omega * t * 0.5)
    field = torch.sin(torch.linspace(0.0, 6.0 * math.pi, series_length, dtype=torch.float32))

    spectral = compute_spectral_signature(signal, dt=dt)
    # Largest structure-function lag min(cap, max(1, n // d)) with cap and d from
    # computation.diagnostics.dynamical_benchmark.structure_function_max_lag / _lag_divisor.
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    lag_cap = int(manager.get(
        "computation.diagnostics.dynamical_benchmark.structure_function_max_lag", required=True
    ))
    lag_divisor = int(manager.get(
        "computation.diagnostics.dynamical_benchmark.structure_function_lag_divisor", required=True
    ))
    structure = compute_structure_function(
        field, max_lag=min(lag_cap, max(series_length // lag_divisor, 1)), order=2
    )
    embedding_a = compute_dynamical_embedding(
        {"signal": signal, "auxiliary": auxiliary},
        n_components=3,
    )
    embedding_b = compute_dynamical_embedding(
        {"signal": signal * 1.01, "auxiliary": auxiliary},
        n_components=3,
    )
    similarity_score = compute_mode_similarity_score(embedding_a, embedding_b)

    results = {
        "spectral_slope": float(spectral["spectral_slope"].item()),
        "dominant_frequency": float(spectral["dominant_frequency"].item()),
        "scaling_exponent": float(structure["scaling_exponent"].item()),
        "embedding_variance_ratio": embedding_a["explained_variance_ratio"].tolist(),
        "mode_similarity_score": float(similarity_score),
    }
    print_info(f"Dominant frequency: {results['dominant_frequency']:.6e} Hz")
    print_info(f"Spectral slope: {results['spectral_slope']:.6f}")
    print_info(f"Structure scaling exponent: {results['scaling_exponent']:.6f}")
    print_info(f"Mode similarity score: {results['mode_similarity_score']:.6f}")

    if args.output:
        save_output(results, args.output, "json")
    return True
