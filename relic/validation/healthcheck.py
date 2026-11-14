"""Core health check logic for RELIC subsystems."""

from __future__ import annotations

import os
import logging
import time
import importlib
from importlib import util as importlib_util
import sys
from typing import Dict, Any, List

from relic.core import device_policy
from relic.utils.imports import module_available

logger = logging.getLogger(__name__)

STATUS_HEALTHY = "HEALTHY"
STATUS_DEGRADED = "DEGRADED"
STATUS_UNHEALTHY = "UNHEALTHY"
STATUS_UNAVAILABLE = "UNAVAILABLE"


class HealthChecker:
    """Unified health checker for RELIC subsystems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.overall_status = STATUS_HEALTHY
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.recommendations: List[str] = []

    def _normalize_status(self, raw_status: Any) -> str:
        """Normalize a subsystem probe status into the shared contract.

        Every probe is RELIC code with a fixed vocabulary (computation
        healthcheck, data I/O HealthStatus, streaming healthcheck, and this
        checker's own "error"/"disabled" entries); a value outside it is a
        contract violation and raises ValueError instead of being mapped.
        """
        if raw_status is None:
            return STATUS_UNAVAILABLE
        status = str(raw_status).strip().lower()
        if status in ("healthy", "ok", "success"):
            return STATUS_HEALTHY
        if status in ("warning", "warn", "degraded"):
            return STATUS_DEGRADED
        if status in ("unhealthy", "critical", "error", "failed", "fail"):
            return STATUS_UNHEALTHY
        if status in ("unavailable", "disabled", "missing"):
            return STATUS_UNAVAILABLE
        raise ValueError(f"Unrecognized subsystem status {raw_status!r}")

    def _serialize_io_report(self, report: Any) -> Any:
        """Convert I/O health report to a JSON-safe dict when needed."""
        if not hasattr(report, "overall_status") or not hasattr(report, "results"):
            return report
        results = {}
        for name, result in report.results.items():
            results[name] = {
                "status": getattr(result.status, "value", str(result.status)),
                "message": result.message,
                "details": result.details,
                "recommendations": result.recommendations,
                "performance_metrics": result.performance_metrics,
            }
        return {
            "overall_status": getattr(report.overall_status, "value", str(report.overall_status)),
            "timestamp": report.timestamp,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "results": results,
        }

    def _is_streaming_enabled(self) -> bool:
        """Return whether experimental streaming checks are enabled."""
        return os.getenv("RELIC_EXPERIMENTAL_STREAMING", "").lower() in ("1", "true")

    def _import_symbols(self, module_path: str, *symbols: str) -> tuple[Any, ...]:
        """Import symbols from a module after find_spec pre-check."""
        if importlib_util.find_spec(module_path) is None:
            raise ImportError(f"Module not found: {module_path}", name=module_path)
        module = importlib.import_module(module_path)
        values = []
        for symbol in symbols:
            if not hasattr(module, symbol):
                raise ImportError(
                    f"Missing symbol '{symbol}' in module '{module_path}'",
                    name=module_path,
                )
            values.append(getattr(module, symbol))
        return tuple(values)

    def _normalize_streaming_status(self, streaming_status: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize streaming health payload with shared status fields."""
        normalized = dict(streaming_status)
        streaming_raw = normalized["overall"]
        normalized["raw_status"] = streaming_raw
        normalized["status"] = self._normalize_status(streaming_raw)
        return normalized

    def _handle_check_error(
        self,
        exc: Exception,
        unavailable_prefix: str,
        failed_prefix: str,
    ) -> Dict[str, Any]:
        """Map import/runtime exceptions to a standardized health response.

        A missing third-party dependency makes the subsystem UNAVAILABLE (the
        recommendation is to install it). An ImportError of RELIC's own code
        (the failing module name is ``relic`` or ``relic.*``) is a code
        defect: the subsystem is UNHEALTHY and the recommendation is to debug.
        """
        if isinstance(exc, ImportError):
            module_name = exc.name or ""
            if module_name == "relic" or module_name.startswith("relic."):
                error_msg = f"{failed_prefix}: RELIC module import failed (code defect): {exc}"
                self.errors.append(error_msg)
                return {"status": STATUS_UNHEALTHY, "error": error_msg}
            error_msg = f"{unavailable_prefix}: {exc}"
            self.errors.append(error_msg)
            return {"status": STATUS_UNAVAILABLE, "error": error_msg}
        error_msg = f"{failed_prefix}: {exc}"
        self.errors.append(error_msg)
        return {"status": STATUS_UNHEALTHY, "error": error_msg}

    def _compose_data_health_result(
        self,
        io_summary: Dict[str, Any],
        streaming_status: Dict[str, Any],
        *,
        streaming_enabled: bool,
        detailed: bool,
        io_details: Any = None,
        serialize_details: bool = False,
    ) -> Dict[str, Any]:
        """Compose a normalized data-layer health report."""
        normalized_io = dict(io_summary)
        io_raw_status = normalized_io["overall"]
        io_status = self._normalize_status(io_raw_status)
        normalized_io["status"] = io_status
        normalized_io["raw_status"] = io_raw_status

        normalized_streaming = self._normalize_streaming_status(streaming_status)

        result_status = io_status
        if (
            streaming_enabled
            and normalized_streaming["status"] in (STATUS_UNHEALTHY, STATUS_UNAVAILABLE)
            and io_status == STATUS_HEALTHY
        ):
            result_status = STATUS_DEGRADED

        result = {
            "status": result_status,
            "io": normalized_io,
            "streaming": normalized_streaming,
        }
        if detailed and io_details is not None:
            result["details"] = {
                "io": self._serialize_io_report(io_details) if serialize_details else io_details
            }

        if result_status in (STATUS_DEGRADED, STATUS_UNHEALTHY, STATUS_UNAVAILABLE):
            self.warnings.append("Data I/O layer requires attention")
        return result

    async def _check_data_health_async(
        self,
        detailed: bool = False,
        serialize_details: bool = False,
    ) -> Dict[str, Any]:
        """Check data I/O layer health using async streaming path."""
        try:
            io_healthcheck, = self._import_symbols("relic.data.io.healthcheck", "io_healthcheck")
            streaming_healthcheck, = self._import_symbols(
                "relic.data.streaming.healthcheck",
                "streaming_healthcheck",
            )

            io_summary = io_healthcheck(detailed=False)
            streaming_enabled = self._is_streaming_enabled()
            if streaming_enabled:
                try:
                    streaming_status = await streaming_healthcheck(detailed=detailed)
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                    streaming_status = {"overall": "error", "error": str(exc)}
            else:
                streaming_status = {
                    "overall": "disabled",
                    "message": "Experimental streaming not enabled",
                }

            io_details = None
            if detailed:
                io_details = io_healthcheck(detailed=True)
            return self._compose_data_health_result(
                io_summary=io_summary,
                streaming_status=streaming_status,
                streaming_enabled=streaming_enabled,
                detailed=detailed,
                io_details=io_details,
                serialize_details=serialize_details,
            )

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Data layer not available",
                "Data health check failed",
            )

    async def check_all_async(
        self,
        detailed: bool = False,
        benchmark: bool = False,
        serialize_details: bool = False,
    ) -> Dict[str, Any]:
        """Perform complete system health check without CLI output."""
        self.results = {
            "compute": self._check_compute_health(detailed),
            "data": await self._check_data_health_async(
                detailed=detailed,
                serialize_details=serialize_details,
            ),
            "params": self._check_params_health(detailed),
            "optimize": self._check_optimize_health(detailed),
            "stochastic": self._check_stochastic_health(detailed),
            "orchestration": self._check_orchestration_health(detailed),
            "config": self._check_config_health(detailed),
        }

        if benchmark:
            self.results["benchmark"] = self._run_basic_benchmark()

        self._aggregate_results()

        return {
            "overall_status": self.overall_status,
            "subsystems": self.results,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "timestamp": time.time(),
        }

    async def check_subsystem_async(
        self,
        subsystem: str,
        detailed: bool = False,
        serialize_details: bool = False,
    ) -> Dict[str, Any]:
        """Run a single subsystem check for async callers."""
        if subsystem == "compute":
            return self._check_compute_health(detailed)
        if subsystem == "data":
            return await self._check_data_health_async(
                detailed=detailed,
                serialize_details=serialize_details,
            )
        if subsystem == "params":
            return self._check_params_health(detailed)
        if subsystem == "orchestration":
            return self._check_orchestration_health(detailed)
        if subsystem == "optimize":
            return self._check_optimize_health(detailed)
        if subsystem == "stochastic":
            return self._check_stochastic_health(detailed)
        if subsystem == "config":
            return self._check_config_health(detailed)
        return {"status": STATUS_UNAVAILABLE, "error": f"Unknown subsystem: {subsystem}"}

    def check_all(self, detailed: bool = False, benchmark: bool = False) -> Dict[str, Any]:
        """Perform complete system health check without CLI output."""
        self.results = {
            "compute": self._check_compute_health(detailed),
            "data": self._check_data_health(detailed),
            "params": self._check_params_health(detailed),
            "optimize": self._check_optimize_health(detailed),
            "stochastic": self._check_stochastic_health(detailed),
            "config": self._check_config_health(detailed),
            "orchestration": self._check_orchestration_health(detailed),
        }

        if benchmark:
            self.results["benchmark"] = self._run_basic_benchmark()

        self._aggregate_results()

        return {
            "overall_status": self.overall_status,
            "subsystems": self.results,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "timestamp": time.time(),
        }

    def _check_compute_health(self, detailed: bool = False) -> Dict[str, Any]:
        """Check computational optimization layer health."""
        try:
            healthcheck, get_device_info = self._import_symbols(
                "relic.computation",
                "healthcheck",
                "get_device_info",
            )

            status = healthcheck(verbose=detailed)
            raw_status = status["overall"]
            normalized_status = self._normalize_status(raw_status)

            result = {
                "status": normalized_status,
                "raw_status": raw_status,
                "dependencies": status.get("dependencies", {}),
                "details": {},
            }

            if detailed:
                result["details"]["device_info"] = get_device_info()
                result["details"]["capabilities"] = status.get("capabilities", {})
                result["details"]["warnings"] = status.get("warnings", [])
                result["details"]["recommendations"] = status.get("recommendations", [])

            if normalized_status == STATUS_DEGRADED:
                self.warnings.append("Computation layer is degraded")
            elif normalized_status in (STATUS_UNHEALTHY, STATUS_UNAVAILABLE):
                self.errors.append("Computation layer is unhealthy")

            return result

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Computation layer not available",
                "Computation health check failed",
            )

    def _check_data_health(self, detailed: bool = False) -> Dict[str, Any]:
        """Check data I/O layer health."""
        try:
            io_healthcheck, = self._import_symbols("relic.data.io.healthcheck", "io_healthcheck")
            streaming_healthcheck_sync, = self._import_symbols(
                "relic.data.streaming.healthcheck",
                "streaming_healthcheck_sync",
            )

            io_summary = io_healthcheck(detailed=False)
            streaming_enabled = self._is_streaming_enabled()
            if streaming_enabled:
                try:
                    streaming_status = streaming_healthcheck_sync(detailed=detailed)
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                    streaming_status = {"overall": "error", "error": str(exc)}
            else:
                streaming_status = {"overall": "disabled", "message": "Experimental streaming not enabled"}

            io_details = None
            if detailed:
                io_details = io_healthcheck(detailed=True)
            return self._compose_data_health_result(
                io_summary=io_summary,
                streaming_status=streaming_status,
                streaming_enabled=streaming_enabled,
                detailed=detailed,
                io_details=io_details,
                serialize_details=False,
            )

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Data layer not available",
                "Data health check failed",
            )

    def _check_params_health(self, detailed: bool = False) -> Dict[str, Any]:
        """Check parameter system health."""
        try:
            get_global_manager, = self._import_symbols(
                "relic.parameters.manager",
                "get_global_manager",
            )

            manager = get_global_manager(preload_all=True)
            param_keys = manager.list()
            namespaces = {
                key.rsplit(".", 1)[0]
                for key in param_keys
                if isinstance(key, str) and "." in key
            }

            result = {
                "status": "healthy",
                "namespaces": len(namespaces),
                "total_parameters": len(param_keys),
                "validation_status": "skipped",
            }

            if detailed:
                try:
                    validation_errors = manager.validate_parameters()
                    result["validation_details"] = validation_errors
                    if validation_errors:
                        result["status"] = "degraded"
                        result["validation_status"] = "failed"
                        self.warnings.append("Some parameters failed validation")
                    else:
                        result["validation_status"] = "passed"
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                    # Health boundary: parameter validation that could not run
                    # leaves the parameter system unverified, so the subsystem
                    # is unhealthy (a validation that ran and found errors is
                    # degraded above).
                    result["status"] = "unhealthy"
                    result["validation_status"] = "error"
                    result["validation_error"] = str(exc)
                    self.errors.append(f"Parameter validation failed: {exc}")

            result["status"] = self._normalize_status(result["status"])
            return result

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Parameter system not available",
                "Parameter health check failed",
            )

    def _check_orchestration_health(self, detailed: bool = False) -> Dict[str, Any]:
        """Check graph orchestration health."""
        try:
            SSOTCatalog, = self._import_symbols(
                "relic.orchestration.ssot_catalog",
                "SSOTCatalog",
            )
            NodeRegistry, = self._import_symbols(
                "relic.orchestration.node_registry",
                "NodeRegistry",
            )
            AdapterRegistry, = self._import_symbols(
                "relic.orchestration.adapters",
                "AdapterRegistry",
            )

            catalog = SSOTCatalog.from_parameter_manager()
            registry = NodeRegistry(ssot_catalog=catalog)
            adapter_registry = AdapterRegistry()

            missing_adapters = [
                name for name in ("unit_cast", "temporal_resample", "semantic_adapter")
                if adapter_registry.get(name) is None
            ]

            # SSOTCatalog loading raises on any entry, standard_name or alias conflict,
            # so a loaded catalog is conflict-free by construction.
            status = STATUS_HEALTHY
            if missing_adapters:
                status = STATUS_DEGRADED
                self.warnings.append("Missing adapters in orchestration registry")

            result = {
                "status": status,
                "missing_adapters": missing_adapters,
            }

            if detailed:
                result["details"] = {
                    "registry_initialized": registry is not None,
                }

            return result

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Orchestration layer not available",
                "Orchestration health check failed",
            )

    def _check_optimize_health(self, detailed: bool = False) -> Dict[str, Any]:
        """Check optimization algorithms health."""
        try:
            get_available_optimizers, = self._import_symbols(
                "relic.optimization",
                "get_available_optimizers",
            )

            optimizers = get_available_optimizers()
            availability: Dict[str, bool] = {}
            available_optimizers = []
            missing_optimizers: Dict[str, Any] = {}
            blocked_optimizers = []
            unclassified_optimizers = []

            for name, info in optimizers.items():
                component_available = info.get("component_available")
                if type(component_available) is not bool:
                    availability[name] = False
                    unclassified_optimizers.append(name)
                    continue

                availability[name] = component_available
                if not component_available:
                    error = info.get("error")
                    missing_optimizers[name] = error
                    if not isinstance(error, str) or not error.strip():
                        unclassified_optimizers.append(name)
                    continue

                available_optimizers.append(name)
                is_blocked_contract = (
                    info.get("error") is None
                    and info.get("runtime_ready") is False
                    and info.get("status") == "[N]"
                    and info.get("contract") == "algorithm_component_only"
                )
                if is_blocked_contract:
                    blocked_optimizers.append(name)
                else:
                    unclassified_optimizers.append(name)

            ready_optimizers: list[str] = []
            runtime_ready = False

            if not availability or (
                not available_optimizers and not unclassified_optimizers
            ):
                status = STATUS_UNAVAILABLE
            else:
                status = STATUS_DEGRADED

            result = {
                "status": status,
                "available_optimizers": available_optimizers,
                "subsystems": availability,
                "runtime": {
                    "status": "[I]" if runtime_ready else "[N]",
                    "ready": runtime_ready,
                    "ready_optimizers": ready_optimizers,
                    "blocked_optimizers": blocked_optimizers,
                    "unclassified_optimizers": unclassified_optimizers,
                },
                "details": optimizers,
                "missing": missing_optimizers,
            }

            if missing_optimizers:
                self.warnings.append("Some optimization subsystems are unavailable")
            if blocked_optimizers or unclassified_optimizers or not available_optimizers:
                self.warnings.append(
                    "Some optimization components are not production-runtime-ready"
                )

            return result

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Optimization layer not available",
                "Optimization health check failed",
            )

    def _check_stochastic_health(self, detailed: bool = False) -> Dict[str, Any]:
        """Check stochastic processes health."""
        try:
            get_available_distributions, get_available_samplers = self._import_symbols(
                "relic.stochastic",
                "get_available_distributions",
                "get_available_samplers",
            )

            distributions = get_available_distributions()
            samplers = get_available_samplers()
            distribution_availability = {
                name: info.get("available", False)
                for name, info in distributions.items()
            }
            sampler_availability = {
                name: info.get("available", False)
                for name, info in samplers.items()
            }
            available_distributions = [
                name for name, info in distributions.items()
                if info.get("available", False)
            ]
            available_samplers = [
                name for name, info in samplers.items()
                if info.get("available", False)
            ]

            ready_samplers = [
                name
                for name, info in samplers.items()
                if info.get("available", False)
                and info.get("runtime_ready") is True
            ]
            blocked_samplers = [
                name
                for name, info in samplers.items()
                if info.get("available", False)
                and info.get("runtime_ready") is False
            ]
            unclassified_samplers = [
                name
                for name, info in samplers.items()
                if info.get("available", False)
                and info.get("runtime_ready") is not True
                and info.get("runtime_ready") is not False
            ]
            runtime_ready = bool(available_samplers) and not (
                blocked_samplers or unclassified_samplers
            )
            unavailable_components = any(
                not available
                for available in (
                    *distribution_availability.values(),
                    *sampler_availability.values(),
                )
            )

            if not available_distributions and not available_samplers:
                status = STATUS_UNAVAILABLE
            elif unavailable_components or not runtime_ready:
                status = STATUS_DEGRADED
            else:
                status = STATUS_HEALTHY

            result = {
                "status": status,
                "distributions": available_distributions,
                "samplers": available_samplers,
                "components": sampler_availability,
                "runtime": {
                    "status": "[I]" if runtime_ready else "[N]",
                    "ready": runtime_ready,
                    "ready_samplers": ready_samplers,
                    "blocked_samplers": blocked_samplers,
                    "unclassified_samplers": unclassified_samplers,
                },
                "details": {
                    "distributions": distributions,
                    "samplers": samplers,
                },
            }

            if unavailable_components or status == STATUS_UNAVAILABLE:
                self.warnings.append("Some stochastic components are unavailable")
            if blocked_samplers or unclassified_samplers or not available_samplers:
                self.warnings.append(
                    "Some stochastic samplers are not production-runtime-ready"
                )

            return result

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Stochastic layer not available",
                "Stochastic health check failed",
            )

    def _check_config_health(self, detailed: bool = False) -> Dict[str, Any]:
        """Check configuration system health."""
        try:
            get_global_config, validate_configuration = self._import_symbols(
                "relic.core.config",
                "get_global_config",
                "validate_configuration",
            )

            config = get_global_config()
            validation_result = validate_configuration(config)
            config_dict = config.model_dump() if hasattr(config, "model_dump") else getattr(config, "__dict__", {})
            config_source = getattr(config, "_source", None)
            if config_source is None:
                config_source = config_dict.get("_source", "default")

            result = {
                "status": "healthy" if validation_result.is_valid else "degraded",
                "config_source": config_source,
                "sections": list(config_dict.keys()),
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "num_errors": validation_result.num_errors,
                    "num_warnings": validation_result.num_warnings,
                    "num_critical": validation_result.num_critical,
                },
            }

            if not validation_result.is_valid:
                result["validation"]["messages"] = [str(msg) for msg in validation_result.messages]
                if validation_result.num_critical > 0:
                    result["status"] = "unhealthy"
                    self.errors.extend([str(msg) for msg in validation_result.messages])
            result["status"] = self._normalize_status(result["status"])
            return result

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            return self._handle_check_error(
                exc,
                "Configuration system not available",
                "Configuration health check failed",
            )

    def _run_basic_benchmark(self) -> Dict[str, Any]:
        """Run basic performance benchmark.

        No configuration schema declares ``enable_gpu_benchmark``, so every
        benchmark request states it explicitly in the health configuration
        (the CLI ``--gpu-benchmark``/``--no-gpu-benchmark`` flags, the API
        ``gpu_benchmark`` query parameter): a missing key raises KeyError and a
        non-boolean value raises TypeError.
        """
        if "enable_gpu_benchmark" not in self.config:
            raise KeyError(
                "A health benchmark requires the explicit boolean health "
                "configuration key 'enable_gpu_benchmark'."
            )
        enable_gpu_benchmark = self.config["enable_gpu_benchmark"]
        if not isinstance(enable_gpu_benchmark, bool):
            raise TypeError(
                "Health configuration key 'enable_gpu_benchmark' must be a boolean, "
                f"got {type(enable_gpu_benchmark).__name__}."
            )
        if not module_available("numpy"):
            return {"status": STATUS_UNAVAILABLE, "error": "numpy not available"}

        import numpy as np

        start = time.perf_counter()
        # Use elementwise arithmetic benchmark to avoid backend-specific BLAS crashes.
        data = np.random.randn(2_000_000).astype(np.float32)
        _ = float(np.sum(data * data))
        cpu_time = time.perf_counter() - start

        benchmark_result = {
            "status": "completed",
            "cpu_benchmark": {
                "matrix_multiply_time": cpu_time,
                "operations_per_second": (2_000_000 * 2) / cpu_time,
            },
        }

        if not module_available("torch"):
            benchmark_result["gpu_benchmark"] = {"status": STATUS_UNAVAILABLE, "error": "torch not available"}
            return benchmark_result

        import torch

        # GPU benchmark is opt-in because some environments expose CUDA
        # but cannot safely run large GEMM benchmarks in health checks.
        if not enable_gpu_benchmark:
            benchmark_result["gpu_benchmark"] = {
                "status": "skipped",
                "reason": "enable_gpu_benchmark is false",
            }
            return benchmark_result

        if not device_policy.is_cuda_available():
            benchmark_result["gpu_benchmark"] = {
                "status": STATUS_UNAVAILABLE,
                "error": "cuda not available",
            }
            return benchmark_result

        device = device_policy.resolve_device(
            explicit_device="cuda",
            module_name=__name__,
            require_gpu=True,
        )
        torch_data = torch.randn(1000, 1000, device=device)
        torch.cuda.synchronize()

        start = time.perf_counter()
        _ = torch.mm(torch_data, torch_data.T)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        benchmark_result["gpu_benchmark"] = {
            "matrix_multiply_time": gpu_time,
            "speedup_over_cpu": cpu_time / gpu_time,
            "operations_per_second": (1000 * 1000 * 1000) / gpu_time,
        }
        return benchmark_result

    def run_system_benchmark(self) -> Dict[str, Any]:
        """Run benchmark with basic system information and timing."""
        benchmark_start = time.time()
        benchmark_results = self._run_basic_benchmark()

        memory_info: Dict[str, Any]
        psutil_module = None
        if "psutil" in sys.modules:
            psutil_module = sys.modules["psutil"]
        else:
            try:
                if importlib_util.find_spec("psutil") is not None:
                    psutil_module = importlib.import_module("psutil")
            except ValueError:
                # Some tests monkeypatch psutil without __spec__.
                psutil_module = sys.modules.get("psutil")

        if psutil_module is None:
            memory_info = {"error": "psutil not available"}
        else:
            memory = psutil_module.virtual_memory()
            memory_info = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "usage_percent": memory.percent,
            }

        total_time = time.time() - benchmark_start

        return {
            "benchmark_results": benchmark_results,
            "memory_info": memory_info,
            "total_time": total_time,
            "timestamp": time.time(),
        }

    def _aggregate_results(self) -> None:
        """Aggregate individual results into overall status."""
        unhealthy_count = 0
        degraded_count = 0

        # Every subsystem result is built by this checker with a status.
        for _, result in self.results.items():
            status = result["status"]
            if status in [STATUS_UNHEALTHY, STATUS_UNAVAILABLE]:
                unhealthy_count += 1
            elif status == STATUS_DEGRADED:
                degraded_count += 1

        if unhealthy_count > 0:
            self.overall_status = STATUS_UNHEALTHY
        elif degraded_count > 0:
            self.overall_status = STATUS_DEGRADED
        else:
            self.overall_status = STATUS_HEALTHY

        self._generate_recommendations()

    def _generate_recommendations(self) -> None:
        """Generate actionable recommendations based on health check results."""
        if self.overall_status == STATUS_UNHEALTHY:
            self.recommendations.append(
                "Critical issues found. Please address errors before using RELIC."
            )

        for subsystem, result in self.results.items():
            status = result["status"]
            if status == STATUS_UNAVAILABLE:
                self.recommendations.append(
                    f"Install missing dependencies for {subsystem} subsystem"
                )
            elif status == STATUS_UNHEALTHY:
                self.recommendations.append(
                    f"Debug errors in {subsystem} subsystem"
                )
            elif status == STATUS_DEGRADED:
                self.recommendations.append(
                    f"Review warnings in {subsystem} subsystem"
                )

        if "benchmark" in self.results:
            benchmark = self.results["benchmark"]
            if "gpu_benchmark" in benchmark and "error" in benchmark["gpu_benchmark"]:
                self.recommendations.append(
                    "Consider installing/configuring GPU support for better performance"
                )
