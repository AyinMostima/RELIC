import asyncio
import importlib
from pathlib import Path

import pytest
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8-sig", errors="ignore")


def test_runtime_red_flags_are_removed_l1() -> None:
    writer_text = _read("relic/data/io/writer.py")
    web_text = _read("relic/interfaces/web/app.py")
    audit_text = _read("relic/coupling/core/audit.py")

    assert "logger = logging.getLogger(__name__)" in writer_text
    assert "get_overall_health(" not in web_text
    assert "build_health_payload(HealthChecker, health_config, detailed=True)" in web_text
    assert "relic.parameters.ops.manager_registry" not in audit_text


def test_helper_deduplication_guardrails_l1() -> None:
    data_cli_text = _read("relic/data/cli_ops.py")
    surrogates_text = _read("relic/models/surrogates.py")
    pytorch_surrogate_text = _read("relic/models/pytorch_surrogate.py")

    assert "def _module_exists" not in data_cli_text
    assert "def _split_names(" not in data_cli_text
    assert not (REPO_ROOT / "relic/optimization/cli_ops.py").exists()
    assert "def _coerce_config(" not in surrogates_text
    assert "def _coerce_settings(" not in pytorch_surrogate_text
    assert "from .config_utils import coerce_config_dict" in surrogates_text
    assert "from .config_utils import coerce_config_dict" in pytorch_surrogate_text


def test_anthroposphere_package_does_not_export_inventory_facades_l1() -> None:
    import relic.scientific.anthroposphere as anthroposphere

    forbidden = {
        "AnthroposphereInventory",
        "InventoryRecord",
        "AggregatedInventory",
        "AggregationPeriod",
        "InventoryCategory",
        "ReportingStandard",
        "EmissionSpecies",
        "EmissionSector",
        "EmissionData",
        "SpatialProxy",
        "VerticalProfile",
    }

    assert "EmissionInventory" in anthroposphere.__all__
    assert forbidden.isdisjoint(anthroposphere.__all__)
    assert not [name for name in forbidden if hasattr(anthroposphere, name)]


def test_emissions_package_does_not_export_inventory_data_schemas_l1() -> None:
    import relic.scientific.anthroposphere.emissions as emissions

    forbidden = {
        "EmissionSpecies",
        "EmissionSector",
        "EmissionData",
        "SpatialProxy",
        "VerticalProfile",
    }

    assert "EmissionInventory" in emissions.__all__
    assert forbidden.isdisjoint(emissions.__all__)
    assert not [name for name in forbidden if hasattr(emissions, name)]


def test_computation_package_has_no_duplicate_apply_compile_entry_l1() -> None:
    import relic.computation as computation

    benchmark_text = _read("scripts/benchmark_computation.py")
    assert not hasattr(computation, "apply_compile")
    assert "apply_compile" not in computation.__all__
    with pytest.raises(ImportError):
        exec("from relic.computation import apply_compile", {})
    assert "apply_compile" not in benchmark_text


def test_jit_package_exposes_only_explicit_compile_entry_l1() -> None:
    import relic.computation.jit as jit_package

    forbidden = {
        "CompilationMode",
        "JITCompiler",
        "CompilationConfig",
        "CompilationCache",
        "CompilationBackend",
        "JITCompilerError",
        "enable_jit",
        "disable_jit",
        "compilation_context",
        "jit_compile_decorator",
        "get_compilation_stats",
        "clear_compilation_cache",
        "jit_healthcheck",
        "compile_scientific_kernel",
        "compile_for_inference",
        "compile_with_dynamic_shapes",
        "_default_compiler",
        "JIT_AVAILABLE",
    }

    assert set(jit_package.__all__) == {"maybe_compile"}
    assert not [name for name in forbidden if hasattr(jit_package, name)]


def test_computation_package_does_not_reexport_duplicate_jit_mode_l1() -> None:
    import relic.computation as computation

    benchmark_text = _read("scripts/benchmark_computation.py")
    assert "CompilationMode" not in computation.__all__
    assert not hasattr(computation, "CompilationMode")
    assert "CompilationMode" not in benchmark_text


def test_jit_compile_module_has_no_global_compiler_or_cache_l1() -> None:
    import relic.computation.jit.compile as compile_module

    forbidden = (
        "CompilationMode",
        "CompilationConfig",
        "CompilationCache",
        "JITCompiler",
        "_compilation_cache",
        "_default_compiler",
        "_COMPILATION_STATS",
        "enable_jit",
        "disable_jit",
        "compilation_context",
        "jit_compile_decorator",
        "get_compilation_stats",
        "clear_compilation_cache",
        "jit_healthcheck",
    )
    assert not [name for name in forbidden if hasattr(compile_module, name)]
    assert compile_module.__all__ == ["maybe_compile"]


def test_jit_adapter_has_no_closure_cross_hit_and_forwards_options_l1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import functools

    import torch

    import relic.computation.jit.compile as compile_module

    compile_calls = []

    def fake_compile(func, **kwargs):
        compile_calls.append(dict(kwargs))

        @functools.wraps(func)
        def compiled(*args, **inner_kwargs):
            return func(*args, **inner_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    def make_function(coefficient):
        def same_code(x):
            return coefficient * x

        return same_code

    compiled2 = compile_module.maybe_compile(
        make_function(2.0),
        backend="eager",
        fullgraph=True,
        dynamic=True,
        compile_mode="reduce-overhead",
        options={"trace.enabled": False},
    )
    compiled3 = compile_module.maybe_compile(
        make_function(3.0),
        backend="aot_eager",
    )

    x = torch.tensor(1.0)
    assert compiled2 is not compiled3
    assert compiled2(x).item() == 2.0
    assert compiled3(x).item() == 3.0
    assert compile_calls[0] == {
        "backend": "eager",
        "fullgraph": True,
        "dynamic": True,
        "mode": "reduce-overhead",
        "options": {"trace.enabled": False},
    }
    assert compile_calls[1]["backend"] == "aot_eager"


def test_jit_adapter_off_and_active_failure_are_fail_closed_l1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import relic.computation.jit.compile as compile_module

    effects = []

    def target(x):
        effects.append(x)
        return x

    assert compile_module.maybe_compile(
        target,
        mode="off",
        allow_uncompiled=True,
    ) is target
    with pytest.raises(RuntimeError, match="disabled"):
        compile_module.maybe_compile(target, mode="off")
    assert effects == []

    def fail_compile(*args, **kwargs):
        raise RuntimeError("backend failure")

    monkeypatch.setattr(torch, "compile", fail_compile)
    with pytest.raises(RuntimeError, match="backend failure"):
        compile_module.maybe_compile(
            target,
            mode="compile",
            allow_uncompiled=True,
        )
    assert effects == []


def test_jit_scientific_validation_is_n_before_compile_or_rng_effect_l1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import relic.computation.jit.compile as compile_module

    effects = []
    compile_calls = []

    def stochastic_target(x):
        effects.append(True)
        return x + torch.rand_like(x)

    monkeypatch.setattr(
        torch,
        "compile",
        lambda *args, **kwargs: compile_calls.append((args, kwargs)),
    )
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match=r"\[N\].*not implemented"):
        compile_module.maybe_compile(
            stochastic_target,
            validate_numerics=True,
            example_inputs=(torch.ones(1),),
        )

    assert compile_calls == []
    assert effects == []
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    input_tensor = torch.ones(2)
    input_alias = input_tensor
    version_before = input_tensor._version

    def input_mutating_target(value):
        value.add_(1.0)
        return value

    with pytest.raises(RuntimeError, match=r"\[N\].*not implemented"):
        compile_module.maybe_compile(
            input_mutating_target,
            validate_numerics=True,
            example_inputs=(input_tensor,),
        )

    assert input_alias is input_tensor
    assert input_tensor._version == version_before
    assert torch.equal(input_tensor, torch.ones_like(input_tensor))


def test_optimization_root_does_not_export_duplicate_runner_l1() -> None:
    import relic.optimization as optimization

    optimization_text = _read("relic/optimization/__init__.py")
    assert "RunnerAdapter" not in optimization.__all__
    assert getattr(optimization, "RunnerAdapter", None) is None
    assert '"RunnerAdapter"' not in optimization_text
    assert "'RunnerAdapter'" not in optimization_text


def test_optimization_calibration_namespace_has_no_runtime_facade_l1() -> None:
    import relic.optimization.calibration as optimization_calibration
    from relic.parameters.calibration.calibrator import ParameterCalibrator

    calibration_text = _read("relic/optimization/calibration/__init__.py")
    assert optimization_calibration.ParameterCalibrator is ParameterCalibrator
    assert "RunnerAdapter" not in optimization_calibration.__all__
    assert "OptimizationCalibrator" not in optimization_calibration.__all__
    assert getattr(optimization_calibration, "RunnerAdapter", None) is None
    assert getattr(optimization_calibration, "OptimizationCalibrator", None) is None
    assert ".runner_adapter" not in calibration_text
    assert ".optimizer" not in calibration_text
    assert "__facade_target__" not in calibration_text


def test_duplicate_optimization_calibrator_module_is_removed_l1() -> None:
    assert not (
        REPO_ROOT / "relic/optimization/calibration/optimizer.py"
    ).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("relic.optimization.calibration.optimizer")


def test_duplicate_optimization_runner_adapter_module_is_removed_l1() -> None:
    assert not (
        REPO_ROOT / "relic/optimization/calibration/runner_adapter.py"
    ).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("relic.optimization.calibration.runner_adapter")


def test_unbound_optimization_manager_module_is_removed_l1() -> None:
    assert not (REPO_ROOT / "relic/optimization/manager.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("relic.optimization.manager")


def test_parameter_import_and_units_cleanup_guardrails_l1() -> None:
    params_init_text = _read("relic/parameters/__init__.py")
    manager_text = _read("relic/parameters/manager.py")
    parameter_validators_init_text = _read("relic/parameters/validators/__init__.py")
    parameter_unit_validator_text = _read(
        "relic/parameters/validators/unit_validator.py"
    )
    core_validator_text = _read("relic/core/units/validator.py")
    unit_converter_text = _read("relic/data/standards/unit_converter.py")
    # test/audit/parameter_yaml_audit.py is the single writer of
    # analysis_results/parameter_yaml_audit.json (the G1 owner); the retired duplicate
    # writers audit_parameter_yaml.py and scan_parameter_yaml.py no longer exist.
    parameter_yaml_audit_text = _read("test/audit/parameter_yaml_audit.py")
    scientific_audit_text = _read("test/audit/scientific_audit_scan.py")

    assert "_db_init_success = initialize_database()" not in params_init_text
    assert "def convert_units(" not in manager_text
    assert "def validate_unit(" not in manager_text
    assert "Unit" + "Converter" not in parameter_validators_init_text
    assert "Unit" + "Converter" not in parameter_unit_validator_text
    assert "class UnitValidator" not in core_validator_text
    assert "UnitValidator = StateUnitValidator" not in core_validator_text
    assert "class StateUnitValidator" in core_validator_text
    assert "CFComplianceValidator" not in unit_converter_text
    assert "cf_validator =" not in unit_converter_text
    assert "UNIT_DIM_RULES" not in parameter_yaml_audit_text
    assert "def _guess_dimension(" not in parameter_yaml_audit_text
    assert "is_declared_dimension_compatible" in parameter_yaml_audit_text
    assert not (REPO_ROOT / "test/audit/audit_parameter_yaml.py").exists()
    assert not (REPO_ROOT / "test/audit/scan_parameter_yaml.py").exists()
    assert "DIMENSION_RULES" not in scientific_audit_text
    assert "is_declared_dimension_compatible" in scientific_audit_text
    assert not (REPO_ROOT / "relic/models/units_helper.py").exists()


def test_unbound_optimization_api_fails_closed_l1() -> None:
    from relic.interfaces.api.routes import optimize
    from relic.interfaces.api.schemas import (
        CalibrationRequest,
        ParameterBound,
        SensitivityRequest,
        UncertaintyRequest,
    )

    info = asyncio.run(optimize.optimize_info())
    assert info.status == "warning"
    assert info.data["status"] == "[N]"
    assert info.data["runtime_ready"] is False
    assert info.data["runtime_blocker"] == "runtime_unbound"
    for legacy_key in ("claim_status", "available", "validation_only"):
        assert legacy_key not in info.data
    assert "ready" not in info.message.lower()

    bound = ParameterBound(min=0.0, max=1.0)
    operations = [
        (
            optimize.calibrate_parameters,
            CalibrationRequest(
                parameters=["rate"],
                parameter_bounds={"rate": bound},
                method="differential_evolution",
                max_iterations=10,
                tolerance=1e-6,
            ),
        ),
        (
            optimize.assimilate_data,
            {
                "observation_data": {},
                "observation_operator": "identity",
                "assimilation_window": [0.0, 1.0],
            },
        ),
        (
            optimize.run_sensitivity,
            SensitivityRequest(
                method="morris",
                parameters={"rate": bound},
                n_trajectories=4,
                n_levels=4,
            ),
        ),
        (
            optimize.run_uncertainty_quantification,
            UncertaintyRequest(
                method="ensemble",
                uncertain_parameters={"rate": {}},
                n_samples=8,
                sampling_strategy="lhs",
                confidence_levels=[0.9],
            ),
        ),
    ]
    for operation, payload in operations:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(operation(payload))
        assert exc_info.value.status_code == 501
        assert exc_info.value.detail["status"] == "[N]"
        assert exc_info.value.detail["runtime_ready"] is False
        assert exc_info.value.detail["runtime_blocker"] == "runtime_unbound"
        for legacy_key in ("claim_status", "available", "validation_only"):
            assert legacy_key not in exc_info.value.detail
        assert exc_info.value.detail["code"] == "runtime_unbound"


def test_api_metadata_bounds_unbound_optimization_runtime_l1() -> None:
    from relic.interfaces.api.server import API_DESCRIPTION, create_app

    assert "[N]" in API_DESCRIPTION
    assert "runtime-unbound" in API_DESCRIPTION

    app = create_app(
        enable_cors=False,
        enable_docs=False,
        enable_auth=False,
    )
    root_route = next(route for route in app.routes if route.path == "/")
    response = asyncio.run(root_route.endpoint())

    assert response.status == "success"
    assert response.data["capabilities"]["optimization"] == {
        "status": "[N]",
        "runtime_ready": False,
        "runtime_blocker": "runtime_unbound",
    }


def test_parameter_calibrator_owns_only_derivative_free_optimizers_l1() -> None:
    from unittest.mock import Mock

    from relic.parameters.calibration.calibrator import (
        CalibrationConfig,
        CalibrationMethod,
        ParameterCalibrator,
    )

    calibrator = ParameterCalibrator(parameter_manager=Mock(), hooks=Mock())
    derivative_free_methods = {
        CalibrationMethod.DIFFERENTIAL_EVOLUTION,
        CalibrationMethod.GENETIC_ALGORITHM,
        CalibrationMethod.SIMULATED_ANNEALING,
        CalibrationMethod.PARTICLE_SWARM,
        CalibrationMethod.BAYESIAN_OPTIMIZATION,
    }
    gradient_methods = {
        CalibrationMethod.GRADIENT_DESCENT,
        CalibrationMethod.ADAM,
        CalibrationMethod.LBFGS,
    }

    assert set(calibrator._optimizers) == derivative_free_methods
    assert derivative_free_methods.isdisjoint(gradient_methods)
    assert derivative_free_methods | gradient_methods == set(CalibrationMethod)
    for method in gradient_methods:
        assert CalibrationConfig.create_simple(method=method.value).method is method
    for method_name in (
        "_gradient_descent",
        "_adam_optimizer",
        "_lbfgs_optimizer",
    ):
        assert method_name not in ParameterCalibrator.__dict__


def test_optimization_calibration_facade_documents_split_owners_l1() -> None:
    import relic.optimization.calibration as calibration_facade

    documentation = calibration_facade.__doc__ or ""
    assert "8 optimization algorithms" not in documentation
    assert "five derivative-free" in documentation
    assert "GradientCalibrator" in documentation
    assert not hasattr(calibration_facade, "GradientCalibrator")


def test_parameter_calibration_package_example_uses_canonical_signature_l1() -> None:
    import relic.parameters.calibration as calibration_package

    documentation = calibration_package.__doc__ or ""
    assert "CalibrationConfig(" in documentation
    assert "calibrator.calibrate(config" in documentation
    assert "method='differential_evolution'" not in documentation
    assert "target_parameters=['biosphere.photosynthesis.vcmax25']" not in documentation


def test_computation_package_docs_do_not_claim_scientific_compile_equivalence_l1() -> None:
    import relic.computation as computation_package

    documentation = computation_package.__doc__ or ""
    assert "torch.jit" not in documentation
    assert "Numerical consistency validation across acceleration paths" not in documentation
    assert "generic callable adapter" in documentation
    assert "scientific-kernel approval" in documentation


def test_core_computation_config_exposes_only_live_compile_controls_l1() -> None:
    from relic.core.config import models

    assert not hasattr(models, "JITMode")
    assert "JITMode" not in models.__all__
    fields = models.ComputationConfig.model_fields
    assert "jit_mode" not in fields
    assert "enable_compile" not in fields
    assert {"compile_backend", "compile_mode"}.issubset(fields)


@pytest.mark.parametrize("target", ["photosynthesis", "transport", "decomposition"])
def test_computation_benchmark_kernels_are_explicitly_synthetic_l1(
    target: str,
) -> None:
    from relic.computation.benchmarks import create_benchmark_function

    benchmark = create_benchmark_function(target)
    assert (
        benchmark.relic_benchmark_scope
        == "synthetic_single_environment_microbenchmark"
    )
    assert benchmark.scientific_equivalence is False
    assert benchmark.scientific_runtime_owner is None


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            "gradient_descent",
            id="gradient-descent",
        ),
        pytest.param("adam", id="adam"),
        pytest.param("lbfgs", id="lbfgs"),
    ],
)
def test_parameter_calibrator_rejects_gradient_methods_before_side_effects_l1(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np
    import torch

    from relic.parameters.calibration.calibrator import (
        CalibrationConfig,
        CalibrationMethod,
        ParameterBounds,
        ParameterCalibrator,
    )

    gradient_method = CalibrationMethod(method)
    manager = Mock()
    manager.get.return_value = 0.5
    hooks = Mock()
    calibrator = ParameterCalibrator(parameter_manager=manager, hooks=hooks)
    manager.reset_mock()
    hooks.reset_mock()
    logger = Mock()
    calibrator.logger = logger

    optimizer_dispatch = Mock(
        return_value=SimpleNamespace(
            optimized_parameters={"test.rate": 0.5},
            convergence_history=[],
            iteration_count=0,
            success=False,
            termination_reason="injected trap",
        )
    )
    calibrator._optimizers[gradient_method] = optimizer_dispatch
    progress_path = tmp_path / "forbidden" / "calibration.json"
    config = CalibrationConfig(
        method=gradient_method,
        target_parameters=["test.rate"],
        parameter_bounds={"test.rate": ParameterBounds(0.0, 1.0, "linear")},
        max_iterations=1,
        random_seed=8128,
        save_progress=True,
        progress_file=progress_path,
    )

    objective_calls = []

    def objective(parameters, observations):
        objective_calls.append((parameters, observations))
        raise ValueError("GradientCalibrator objective trap")

    numpy_seed_calls = []
    torch_seed_calls = []
    monkeypatch.setattr(
        np.random,
        "seed",
        lambda *args, **kwargs: numpy_seed_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda *args, **kwargs: torch_seed_calls.append((args, kwargs)),
    )

    numpy_state_before = np.random.get_state()
    torch_state_before = torch.random.get_rng_state().clone()
    cuda_state_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    try:
        with pytest.raises(ValueError, match="GradientCalibrator"):
            calibrator.calibrate(
                config,
                objective,
                observations=object(),
            )
        numpy_state_after = np.random.get_state()
        torch_state_after = torch.random.get_rng_state().clone()
        cuda_state_after = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if cuda_state_before is not None
            else None
        )
    finally:
        np.random.set_state(numpy_state_before)
        torch.random.set_rng_state(torch_state_before)
        if cuda_state_before is not None:
            torch.cuda.set_rng_state_all(cuda_state_before)

    observed_effects = {
        "optimizer_dispatch": optimizer_dispatch.call_count,
        "manager": len(manager.mock_calls),
        "hooks": len(hooks.mock_calls),
        "logger": len(logger.mock_calls),
        "objective": len(objective_calls),
        "numpy_seed": len(numpy_seed_calls),
        "torch_seed": len(torch_seed_calls),
    }
    assert observed_effects == {
        "optimizer_dispatch": 0,
        "manager": 0,
        "hooks": 0,
        "logger": 0,
        "objective": 0,
        "numpy_seed": 0,
        "torch_seed": 0,
    }
    assert numpy_state_after[0] == numpy_state_before[0]
    assert np.array_equal(numpy_state_after[1], numpy_state_before[1])
    assert numpy_state_after[2:] == numpy_state_before[2:]
    assert torch.equal(torch_state_after, torch_state_before)
    if cuda_state_before is not None:
        assert cuda_state_after is not None
        assert len(cuda_state_after) == len(cuda_state_before)
        assert all(
            torch.equal(after, before)
            for after, before in zip(cuda_state_after, cuda_state_before)
        )
    assert not progress_path.exists()


def test_private_multi_isotope_state_runtimes_are_removed_l1() -> None:
    for module_name in ("multi_isotope_tracker", "multi_isotope_pools"):
        assert not (
            REPO_ROOT
            / f"relic/scientific/isotopes/tracking/{module_name}.py"
        ).exists()
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(
                f"relic.scientific.isotopes.tracking.{module_name}"
            )
    assert not (
        REPO_ROOT / "test/audit/repro_multi_isotope_manager.py"
    ).exists()


def test_cross_domain_validator_cf_facade_is_removed_l1() -> None:
    """The package facade must not expose an unregistered CF compliance path."""
    import relic.validation.cross_domain_validator as validator_module
    from relic.validation import CrossDomainValidator

    validator_text = _read("relic/validation/cross_domain_validator.py")

    assert not hasattr(CrossDomainValidator, "validate_cf_compliance")
    assert not hasattr(validator_module, "validate_cf_compliance_impl")
    assert "validate_cf_compliance_impl" not in validator_text
    with pytest.raises(AttributeError):
        getattr(CrossDomainValidator, "validate_cf_compliance")


def test_cross_domain_interface_cf_bridge_is_removed_l1() -> None:
    """Interface helpers must not retain the unregistered CF bridge."""
    import relic.validation.cross_domain_interface_ops as interface_ops
    from relic.validation import CrossDomainValidator

    interface_text = _read("relic/validation/cross_domain_interface_ops.py")

    assert CrossDomainValidator is not None
    assert not hasattr(interface_ops, "validate_cf_compliance_impl")
    assert not hasattr(interface_ops, "framework_validate_cf")
    assert "validate_cf_compliance_impl" not in interface_ops.__all__
    assert "validate_cf_compliance_impl" not in interface_text
    assert "framework_validate_cf" not in interface_text
    with pytest.raises(ImportError):
        exec(
            "from relic.validation.cross_domain_interface_ops "
            "import validate_cf_compliance_impl",
            {},
        )


def test_temporal_metadata_duplicate_module_is_absent_l1() -> None:
    module_name = "relic.coupling.scales.temporal.cf_metadata"

    assert importlib.util.find_spec(module_name) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_coupling_package_has_no_temporal_metadata_manager_binding_l1() -> None:
    import relic.coupling as coupling

    assert not hasattr(coupling, "CFTemporalMetadataManager")
    assert "CFTemporalMetadataManager" not in coupling.__all__


def test_scales_package_has_no_temporal_metadata_manager_binding_l1() -> None:
    import relic.coupling.scales as scales

    assert not hasattr(scales, "CFTemporalMetadataManager")
    assert "CFTemporalMetadataManager" not in scales.__all__


def test_temporal_package_has_no_legacy_metadata_bindings_l1() -> None:
    import relic.coupling.scales.temporal as temporal

    for name in (
        "CFTemporalMetadataManager",
        "CFTimeAttributes",
        "CFQualityFlags",
        "CFHistoryTracker",
        "CFHistoryEntry",
        "update_time_metadata",
    ):
        assert not hasattr(temporal, name)
        assert name not in temporal.__all__


def test_temporal_package_exports_only_bounded_canonical_surface_l1() -> None:
    import relic.coupling.scales.temporal as temporal

    for required in (
        "TemporalScaling",
        "TemporalAggregator",
        "TemporalDisaggregator",
        "TemporalSemantics",
    ):
        assert hasattr(temporal, required)
        assert required in temporal.__all__
    for retired in (
        "infer_variable_role",
        "compute_time_bounds",
        "ShapeFunction",
        "ShapeFunctionType",
        "_TEMPORAL_REEXPORTS",
    ):
        assert not hasattr(temporal, retired)
        assert retired not in temporal.__all__

    contract = (temporal.__doc__ or "").lower()
    for stale_claim in (
        "support for 5 calendar types",
        "advanced aggregation methods",
        "advanced disaggregation methods",
        "full functionality",
        "compatibility",
    ):
        assert stale_claim not in contract
    for bounded_claim in (
        "bounded",
        "interval_mean",
        "rate",
        "interval_accumulation",
        "state_at_boundary",
        "fail closed",
    ):
        assert bounded_claim in contract


def test_scales_package_exports_temporal_facade_without_core_helpers_l1() -> None:
    import relic.coupling.scales as scales

    for required in (
        "TemporalScaling",
        "TemporalAggregator",
        "TemporalDisaggregator",
        "TemporalSemantics",
    ):
        assert hasattr(scales, required)
        assert required in scales.__all__
    for internal in (
        "TimeWeightGenerator",
        "apply_temporal_kernel",
        "compute_temporal_mapping_residual",
    ):
        assert not hasattr(scales, internal)
        assert internal not in scales.__all__


def test_coupling_package_exports_temporal_facade_without_core_helpers_l1() -> None:
    import relic.coupling as coupling

    for required in (
        "TemporalScaling",
        "TemporalAggregator",
        "TemporalDisaggregator",
        "TemporalSemantics",
    ):
        assert hasattr(coupling, required)
        assert required in coupling.__all__
    for internal in (
        "TimeWeightGenerator",
        "apply_temporal_kernel",
        "compute_temporal_mapping_residual",
    ):
        assert not hasattr(coupling, internal)
        assert internal not in coupling.__all__


def test_temporal_core_has_no_role_inference_or_bounds_guessing_l1() -> None:
    import relic.coupling.scales.temporal.core as temporal_core

    for retired in (
        "VARIABLE_ROLES",
        "infer_variable_role",
        "compute_time_bounds",
    ):
        assert not hasattr(temporal_core, retired)
        assert retired not in temporal_core.__all__
    source = _read("relic/coupling/scales/temporal/core.py")
    assert "def infer_variable_role" not in source
    assert "def compute_time_bounds" not in source


def test_coupling_package_has_no_deprecated_cf_adapter_surface_l1() -> None:
    import relic.coupling as coupling

    for retired in ("cf", "CFSynchronizerAdapter", "get_cf_adapter"):
        assert not hasattr(coupling, retired)
        assert retired not in coupling.__all__


def test_deprecated_coupling_cf_package_is_absent_l1() -> None:
    module_name = "relic.coupling.cf"

    assert importlib.util.find_spec(module_name) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_basin_configuration_has_no_unbound_characteristic_time_l1() -> None:
    import relic.models.basin as basin

    assert not hasattr(basin, "CHARACTERISTIC_TIME")
    assert "CHARACTERISTIC_TIME" not in _read("relic/models/basin.py")


def test_hillslope_configuration_has_no_unbound_characteristic_time_l1() -> None:
    import relic.models.hillslope as hillslope

    assert not hasattr(hillslope, "CHARACTERISTIC_TIME")
    assert "CHARACTERISTIC_TIME" not in _read("relic/models/hillslope.py")
