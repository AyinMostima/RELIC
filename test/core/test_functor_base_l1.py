"""
L1 Unit Tests for FunctorBase

Tests individual methods and functionality of FunctorBase class
in isolation, ensuring correct behavior of core operations.
"""

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from types import CodeType, FunctionType, MappingProxyType, MethodType
import torch
from relic.core.functor import (
    FunctorMetadata, ProcessCategory, ExecutionPhase
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import ExternalIntent


class _CallContractFunctor(BiologicalProcessFunctor):
    """Exercise the public call contract with visible lifecycle effects."""

    def __init__(self):
        super().__init__(
            name="call_contract",
            config={},
            metadata=FunctorMetadata(
                name="call_contract",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                description="Call lifecycle test fixture",
                references=["PyTorch Module call and hook contracts"],
                input_variables=["input_state"],
                output_variables=["output_state"],
            ),
            process_name="call_contract",
        )
        self.events = []

    def get_dependencies(self):
        return []

    def preprocess_state(self, state):
        self.events.append("preprocess")
        return super().preprocess_state(state)

    def validate_state(self, state):
        self.events.append("validate_input")
        return super().validate_state(state)

    def forward(self, state, time, dt, **kwargs):
        self.events.append("kernel")
        if kwargs.get("fail"):
            raise RuntimeError("injected kernel failure")
        output = state["input_state"] * 2.0
        if kwargs.get("tensor_output"):
            return output
        return {"output_state": output}

    def postprocess_updates(self, original_state, updates):
        self.events.append("postprocess")
        return {"output_state": updates["output_state"] + 1.0}

    def _check_local_conservation(self, state, updates):
        self.events.append("conservation")
        return super()._check_local_conservation(state, updates)

    def _run_validation_pipeline(self, state, updates, time, dt, validation_report=None):
        self.events.append("validate_output")
        return super()._run_validation_pipeline(state, updates, time, dt, validation_report)


class _IntentCallContractFunctor(_CallContractFunctor):
    def __init__(self, *, from_postprocess=False):
        super().__init__()
        self.from_postprocess = from_postprocess
        self.intent = ExternalIntent("write_outputs", "output_file", "snapshot")
        self.postprocess_keys = set()
        self.conservation_keys = set()
        self.validation_keys = set()

    def forward(self, state, time, dt, **kwargs):
        result = super().forward(state, time, dt, **kwargs)
        if not self.from_postprocess:
            result["export"] = self.intent
        return result

    def postprocess_updates(self, original_state, updates):
        self.postprocess_keys = set(updates)
        result = super().postprocess_updates(original_state, updates)
        if self.from_postprocess:
            result["export"] = self.intent
        return result

    def _check_local_conservation(self, state, updates):
        self.conservation_keys = set(updates)
        return super()._check_local_conservation(state, updates)

    def _run_validation_pipeline(self, state, updates, time, dt, validation_report=None):
        self.validation_keys = set(updates)
        return super()._run_validation_pipeline(state, updates, time, dt, validation_report)


def _identity_default_forward(self, state, time, dt, gain=2.0, *, offset=1.0):
    return {"output_state": state["input_state"] * gain + offset}


@wraps(_identity_default_forward)
def _identity_wrapped_forward(self, state, time, dt, gain=2.0, *, offset=1.0):
    return {"output_state": state["input_state"] * gain - offset}


def _identity_nested_forward(self, state, time, dt):
    def factor():
        return 2.0
    return {"output_state": state["input_state"] * factor()}


def _identity_closure_forward(value):
    def forward(self, state, time, dt):
        return {"output_state": state["input_state"] * value}
    return forward


@contextmanager
def _replace_function_attribute(function, name, value):
    """Restore function descriptors by assignment, never descriptor deletion."""
    original = getattr(function, name)
    setattr(function, name, value)
    try:
        yield
    finally:
        setattr(function, name, original)


class _IdentitySuperFunctor(_CallContractFunctor):
    def forward(self, state, time, dt, **kwargs):
        return super().forward(state, time, dt, **kwargs)


class TestFunctorBaseL1(unittest.TestCase):
    """L1 unit tests for FunctorBase functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_metadata = FunctorMetadata(
            name="test_functor",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            version="1.0.0",
            description="Unit test functor metadata",
            references=["Unit test reference"],
            output_variables=["output_state"],
        )

    def test_metadata_initialization(self):
        """Test FunctorMetadata initialization and validation."""
        # Test valid initialization
        metadata = FunctorMetadata(
            name="test",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.PREPROCESSING,
            description="Unit test metadata",
            references=["Unit test reference"],
            output_variables=["test_output"],
        )
        self.assertEqual(metadata.name, "test")
        self.assertEqual(metadata.category, ProcessCategory.BIOLOGICAL_PROCESS)
        self.assertEqual(metadata.execution_phase, ExecutionPhase.PREPROCESSING)

    def test_metadata_to_dict(self):
        """Test metadata serialization to dictionary."""
        metadata_dict = self.test_metadata.to_dict()
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict['name'], "test_functor")
        self.assertEqual(metadata_dict['category'], "biological_process")
        self.assertEqual(metadata_dict['execution_phase'], "main_processes")

    def test_metadata_serializes_optional_outputs(self):
        """Optional outputs remain members of the single declared output contract."""
        metadata = FunctorMetadata(
            name="conditional_output_test",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Conditional output metadata test",
            references=["Unit test reference"],
            output_variables=["required_output", "conditional_output"],
            optional_output_variables=["conditional_output"],
        )

        self.assertEqual(metadata.optional_output_variables, ["conditional_output"])
        self.assertEqual(
            metadata.to_dict()["optional_output_variables"],
            ["conditional_output"],
        )

    def test_metadata_declares_sub_flow_accuracy(self):
        """Local accuracy of the Functor's own sub-flow; UNDECLARED unless stated."""
        from relic.core.functor.functor_types import SubFlowAccuracy

        import relic.core
        import relic.core.functor

        self.assertIs(relic.core.functor.SubFlowAccuracy, SubFlowAccuracy)
        self.assertIs(relic.core.SubFlowAccuracy, SubFlowAccuracy)
        self.assertIn("SubFlowAccuracy", relic.core.functor.__all__)
        self.assertIn("SubFlowAccuracy", relic.core.__all__)
        self.assertIs(self.test_metadata.sub_flow_accuracy, SubFlowAccuracy.UNDECLARED)
        self.assertEqual(self.test_metadata.to_dict()["sub_flow_accuracy"], "undeclared")
        declared = FunctorMetadata(
            name="exact_sub_flow",
            description="Exact sub-flow metadata test",
            references=["Unit test reference"],
            output_variables=["output_state"],
            sub_flow_accuracy="order_2",
        )
        self.assertIs(declared.sub_flow_accuracy, SubFlowAccuracy.ORDER_2)
        self.assertEqual(declared.to_dict()["sub_flow_accuracy"], "order_2")
        self.assertEqual(SubFlowAccuracy.EXACT.order, float("inf"))
        self.assertEqual(SubFlowAccuracy.ORDER_1.order, 1)
        self.assertIsNone(SubFlowAccuracy.UNDECLARED.order)
        with self.assertRaisesRegex(ValueError, "sub_flow_accuracy"):
            FunctorMetadata(
                name="invalid_sub_flow",
                description="Invalid sub-flow metadata test",
                references=["Unit test reference"],
                output_variables=["output_state"],
                sub_flow_accuracy="order_3",
            )

    def test_metadata_rejects_optional_output_outside_declared_contract(self):
        """An optional output cannot escape the canonical output declaration."""
        with self.assertRaisesRegex(ValueError, "optional_output_variables"):
            FunctorMetadata(
                name="invalid_conditional_output_test",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                description="Invalid conditional output metadata test",
                references=["Unit test reference"],
                output_variables=["required_output"],
                optional_output_variables=["undeclared_output"],
            )

    def test_parameter_access(self):
        """Test parameter access through get_parameter_value."""
        # Create a concrete functor for testing
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="test",
            config={"parameters": {"test_param": 10.0}},
            metadata=self.test_metadata,
            process_name="test_process"
        )

        # The instance config supplies the value (no caller default exists).
        value = functor.get_parameter_value(
            "test_param",
            units="dimensionless"
        )
        self.assertEqual(value, 10.0)
        self.assertIsNone(functor.get_parameter_value("absent_param"))

    def test_conservation_check(self):
        """Test conservation checking logic."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {"carbon_flux": torch.tensor(1.0)}

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="test",
            config={},
            metadata=self.test_metadata,
            process_name="test_process"
        )

        # Test conservation check with valid flux
        state = {"carbon": torch.tensor(100.0)}
        result = functor(state, time=0.0, dt=1.0)
        self.assertIn("carbon_flux", result)

    def test_generic_conservation_does_not_guess_undeclared_outputs(self):
        """The base checker must not infer a ledger from undeclared outputs."""
        metadata = FunctorMetadata(
            name="undeclared_conservation_test",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Undeclared conservation output test",
            references=["Unit test reference"],
            conservation_laws=["mass"],
            input_variables=["input_state"],
        )

        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="undeclared_conservation_test",
            config={},
            metadata=metadata,
            process_name="undeclared_conservation_process",
        )

        self.assertEqual(
            functor.check_conservation_laws(
                {},
                {"unowned_flux": torch.tensor(5.0)},
            ),
            {},
        )

    def test_domain_conservation_checker_receives_declared_diagnostics(self):
        """Domain ledgers receive diagnostics while the generic checker ignores them."""
        metadata = FunctorMetadata(
            name="diagnostic_ledger_functor",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            version="1.0.0",
            description="Domain conservation ledger test",
            references=["Unit test reference"],
            output_variables=[
                "boundary_flux",
                "diagnostics.test.mass_balance_residual",
            ],
            conservation_laws=["mass"],
        )

        class DiagnosticLedgerFunctor(BiologicalProcessFunctor):
            def __init__(self):
                super().__init__(
                    name="diagnostic_ledger_functor",
                    config={},
                    metadata=metadata,
                    process_name="diagnostic_ledger_process",
                )
                self.checked_output_keys = set()

            def forward(self, state, time, dt, **kwargs):
                return {
                    "boundary_flux": torch.tensor(1.0),
                    "diagnostics.test.mass_balance_residual": torch.tensor(0.0),
                    "export": ExternalIntent("export_diagnostics", "diagnostics", "snapshot"),
                }

            def check_conservation_laws(self, input_state, output_updates):
                self.checked_output_keys = set(output_updates)
                residual = output_updates[
                    "diagnostics.test.mass_balance_residual"
                ]
                return {"mass": float(torch.abs(residual).item())}

            def get_dependencies(self):
                return []

        functor = DiagnosticLedgerFunctor()
        result = functor({}, time=0.0, dt=1.0)

        self.assertIn("boundary_flux", result)
        self.assertEqual(
            functor.checked_output_keys,
            {
                "boundary_flux",
                "diagnostics.test.mass_balance_residual",
            },
        )

        functor.checked_output_keys = set()
        functor._enforce_forward_conservation(
            {"boundary_flux": torch.tensor(0.0)},
            result,
        )
        self.assertEqual(
            functor.checked_output_keys,
            {
                "boundary_flux",
                "diagnostics.test.mass_balance_residual",
            },
        )

    def test_execution_timing(self):
        """Test execution timing functionality."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                import time
                time.sleep(0.01)  # Simulate some work
                return {}

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="test",
            config={},
            metadata=self.test_metadata,
            process_name="test_process"
        )

        # Execute and check timing
        state = {}
        functor(state, time=0.0, dt=1.0)

        # Check that execution time was recorded
        self.assertGreater(functor._last_execution_time, 0)
        self.assertGreater(functor._execution_count, 0)

    def test_opt_in_pipeline_timing_separates_validation_and_conservation(self):
        """Opt-in profiling must expose stable Functor pipeline stage timings."""
        expected_keys = {
            "state_validation",
            "forward",
            "local_conservation",
            "framework_validation",
            "total",
        }

        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {"output": torch.tensor(1.0)}

            def get_dependencies(self):
                return []

        disabled = TestFunctor(
            name="pipeline_timing_disabled",
            config={},
            metadata=self.test_metadata,
            process_name="pipeline_timing_disabled_process",
        )
        disabled({}, time=0.0, dt=1.0)
        self.assertEqual(set(disabled._last_pipeline_timings_s), expected_keys)
        self.assertTrue(
            all(value == 0.0 for value in disabled._last_pipeline_timings_s.values())
        )
        self.assertEqual(disabled._pipeline_timing_count, 0)

        profiled = TestFunctor(
            name="pipeline_timing_profiled",
            config={},
            metadata=self.test_metadata,
            process_name="pipeline_timing_profiled_process",
        )
        profiled._performance_profiling_enabled = True
        profiled({}, time=0.0, dt=1.0)

        timings = profiled._last_pipeline_timings_s
        self.assertEqual(set(timings), expected_keys)
        self.assertGreater(timings["total"], 0.0)
        self.assertGreater(timings["state_validation"], 0.0)
        self.assertGreater(timings["forward"], 0.0)
        self.assertGreaterEqual(timings["local_conservation"], 0.0)
        self.assertGreater(timings["framework_validation"], 0.0)
        self.assertGreaterEqual(
            timings["total"],
            timings["state_validation"] + timings["forward"],
        )
        self.assertEqual(profiled._pipeline_timing_count, 1)
        self.assertEqual(set(profiled._pipeline_timing_totals_s), expected_keys)
        for stage_name in expected_keys:
            self.assertGreaterEqual(
                profiled._pipeline_timing_totals_s[stage_name],
                timings[stage_name],
            )

        profiled({}, time=1.0, dt=1.0)
        self.assertEqual(profiled._pipeline_timing_count, 2)

        profiled.reset_stats()
        self.assertEqual(profiled._pipeline_timing_count, 0)
        self.assertTrue(
            all(value == 0.0 for value in profiled._last_pipeline_timings_s.values())
        )
        self.assertTrue(
            all(value == 0.0 for value in profiled._pipeline_timing_totals_s.values())
        )

    def test_call_governor_preserves_raw_kernel_and_lifecycle(self):
        """Governance composes the normal call without replacing forward."""
        functor = _CallContractFunctor()
        raw_forward = functor.forward.__func__

        def govern(call, state, *, time, dt, **kwargs):
            functor.events.append("govern_before")
            result = call(state, time=time, dt=dt, **kwargs)
            functor.events.append("govern_after")
            return result

        functor.set_call_governor(govern)
        state = {"input_state": torch.tensor(3.0)}
        result = functor(state, time=0.0, dt=1.0)

        self.assertEqual(float(result["output_state"]), 7.0)
        self.assertEqual(
            functor.events,
            [
                "preprocess", "validate_input", "govern_before", "kernel",
                "govern_after", "postprocess", "conservation", "validate_output",
            ],
        )
        self.assertIs(functor.forward.__func__, raw_forward)
        self.assertNotIn("forward", vars(functor))
        functor.events.clear()
        self.assertEqual(float(functor.forward(state, 0.0, 1.0)["output_state"]), 6.0)
        self.assertEqual(functor.events, ["kernel"])

    def test_governed_call_keeps_intents_out_of_scientific_views(self):
        for from_postprocess in (False, True):
            with self.subTest(from_postprocess=from_postprocess):
                functor = _IntentCallContractFunctor(from_postprocess=from_postprocess)
                value = torch.tensor(3.0, requires_grad=True)
                result = functor({"input_state": value}, time=0.0, dt=1.0)
                self.assertIs(result["export"], functor.intent)
                self.assertEqual(result["output_state"].item(), 7.0)
                self.assertEqual(functor.postprocess_keys, {"output_state"})
                self.assertEqual(functor.conservation_keys, {"output_state"})
                self.assertEqual(functor.validation_keys, {"output_state"})
                result["output_state"].backward()
                self.assertEqual(value.grad.item(), 2.0)

    def test_external_intent_cannot_be_a_declared_scientific_output(self):
        from relic.core.functor import ExecutionError

        functor = _IntentCallContractFunctor()
        functor.metadata.output_variables.append("export")
        with self.assertRaisesRegex(ExecutionError, "external intent"):
            functor({"input_state": torch.tensor(3.0)}, time=0.0, dt=1.0)

    def test_postprocess_cannot_replace_a_pending_intent_with_state(self):
        from relic.core.functor import ExecutionError

        functor = _IntentCallContractFunctor()
        functor.postprocess_updates = lambda original, updates: {
            "output_state": updates["output_state"], "export": torch.tensor(0.0),
        }
        with self.assertRaisesRegex(ExecutionError, "external intent"):
            functor({"input_state": torch.tensor(3.0)}, time=0.0, dt=1.0)

    def test_call_governor_shortcut_still_validates_and_postprocesses(self):
        """A reused raw result cannot skip input or output governance."""
        from relic.core.functor import ExecutionError

        functor = _CallContractFunctor()

        def govern(call, state, *, time, dt, **kwargs):
            functor.events.append("reuse")
            return {"output_state": torch.tensor(5.0)}

        functor.set_call_governor(govern)
        result = functor({"input_state": torch.tensor(3.0)}, time=0.0, dt=1.0)
        self.assertEqual(float(result["output_state"]), 6.0)
        self.assertEqual(
            functor.events,
            ["preprocess", "validate_input", "reuse", "postprocess",
             "conservation", "validate_output"],
        )
        functor.events.clear()
        with self.assertRaisesRegex(ExecutionError, "Required state variable"):
            functor({}, time=0.0, dt=1.0)
        self.assertEqual(functor.events, ["preprocess", "validate_input"])

    def test_call_governor_binding_validation_and_removal(self):
        """One binding accepts a callable or explicit removal only."""
        functor = _CallContractFunctor()
        for invalid in (False, 1, "govern", {}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                TypeError, "callable or None",
            ):
                functor.set_call_governor(invalid)
        functor.set_call_governor(
            lambda call, state, **kwargs: {"output_state": torch.tensor(9.0)}
        )
        functor.set_call_governor(None)
        result = functor({"input_state": torch.tensor(3.0)}, time=0.0, dt=1.0)
        self.assertEqual(float(result["output_state"]), 7.0)
        self.assertIn("kernel", functor.events)

    def test_call_runs_torch_kwargs_hooks_and_preserves_gradient(self):
        """PyTorch pre/output hooks transform the real differentiable call."""
        functor = _CallContractFunctor()
        value = torch.tensor(3.0, requires_grad=True)

        def before(module, args, kwargs):
            self.assertIs(module, functor)
            self.assertEqual(kwargs["time"], 2.0)
            self.assertEqual(kwargs["dt"], 1.0)
            functor.events.append("torch_pre")
            return ({"input_state": args[0]["input_state"] + 2.0},), kwargs

        def after(module, args, kwargs, output):
            functor.events.append("torch_post")
            return {"output_state": output["output_state"] + 3.0}

        handles = [
            functor.register_forward_pre_hook(before, with_kwargs=True),
            functor.register_forward_hook(after, with_kwargs=True),
        ]
        for handle in handles:
            self.addCleanup(handle.remove)
        result = functor({"input_state": value}, time=2.0, dt=1.0)
        self.assertEqual(float(result["output_state"].detach()), 14.0)
        self.assertEqual(float(torch.autograd.grad(result["output_state"], value)[0]), 2.0)
        self.assertEqual(
            functor.events,
            ["preprocess", "validate_input", "torch_pre", "kernel",
             "torch_post", "postprocess", "conservation", "validate_output"],
        )

    def test_call_runs_global_torch_hooks(self):
        """Global hooks retain native ordering around the kernel."""
        from torch.nn.modules import module as torch_module

        functor = _CallContractFunctor()

        def before(module, args):
            if module is functor:
                functor.events.append("global_pre")

        def after(module, args, output):
            if module is functor:
                functor.events.append("global_post")
                return {"output_state": output["output_state"] * 3.0}

        handles = [
            torch_module.register_module_forward_pre_hook(before),
            torch_module.register_module_forward_hook(after),
        ]
        for handle in handles:
            self.addCleanup(handle.remove)
        result = functor({"input_state": torch.tensor(3.0)}, time=0.0, dt=1.0)
        self.assertEqual(float(result["output_state"]), 19.0)
        self.assertEqual(functor.events[2:5], ["global_pre", "kernel", "global_post"])

    def test_call_runs_always_called_torch_hook_on_kernel_failure(self):
        """Failed kernels run native cleanup hooks and keep the error cause."""
        from relic.core.functor import ExecutionError

        functor = _CallContractFunctor()
        observed = []
        handle = functor.register_forward_hook(
            lambda module, args, output: observed.append(output),
            always_call=True,
        )
        self.addCleanup(handle.remove)
        with self.assertRaisesRegex(ExecutionError, "injected kernel failure") as caught:
            functor({"input_state": torch.tensor(3.0)}, time=0.0, dt=1.0, fail=True)
        self.assertEqual(observed, [None])
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertFalse(functor._inside_canonical_call)
        result = functor({"input_state": torch.tensor(3.0)}, time=0.0, dt=1.0)
        self.assertEqual(float(result["output_state"]), 7.0)

    def test_call_preserves_native_full_backward_hooks_for_tensor_output(self):
        """Tensor outputs retain PyTorch backward pre/post hook behavior."""
        functor = _CallContractFunctor()
        value = torch.tensor(3.0, requires_grad=True)
        events = []

        def before_backward(module, grad_output):
            events.append("backward_pre")
            return tuple(item * 2.0 for item in grad_output)

        def after_backward(module, grad_input, grad_output):
            events.append("backward")
            self.assertEqual(grad_input, (None,))

        handles = [
            functor.register_full_backward_pre_hook(before_backward),
            functor.register_full_backward_hook(after_backward),
        ]
        for handle in handles:
            self.addCleanup(handle.remove)
        report = []
        result = functor(
            {"input_state": value}, time=0.0, dt=1.0, tensor_output=True,
            validation_report=report,
        )
        with self.assertWarnsRegex(UserWarning, "no inputs require gradients"):
            gradient = torch.autograd.grad(result, value)[0]
        self.assertEqual(float(gradient), 4.0)
        self.assertEqual(events, ["backward_pre", "backward"])
        # The bare tensor output passes the (strict) framework validation.
        self.assertEqual([record for record in report if record["errors"]], [])

    def test_call_preserves_native_full_backward_hook_dict_boundary(self):
        """Dict outputs retain the native warning without a second adapter."""
        functor = _CallContractFunctor()
        value = torch.tensor(3.0, requires_grad=True)
        events = []
        handle = functor.register_full_backward_hook(
            lambda module, grad_input, grad_output: events.append("backward")
        )
        self.addCleanup(handle.remove)
        with self.assertWarnsRegex(UserWarning, "module output should be a Tensor"):
            result = functor({"input_state": value}, time=0.0, dt=1.0)
        gradient = torch.autograd.grad(result["output_state"], value)[0]
        self.assertEqual(float(gradient), 2.0)
        self.assertEqual(events, [])

    def test_kernel_identity_is_stable_across_instances_and_governance(self):
        """Only the actual kernel is certified, not instance state or policy."""
        first = _CallContractFunctor()
        second = _CallContractFunctor()
        expected = first.get_kernel_identity()
        second.events.append("different runtime state")
        second.set_call_governor(lambda call, *args, **kwargs: call(*args, **kwargs))
        self.assertEqual(expected, second.get_kernel_identity())
        self.assertEqual(expected["scope"], "bound_python_kernel")

    def test_kernel_identity_changes_for_actual_forward_with_same_class(self):
        functor = _CallContractFunctor()
        expected = functor.get_kernel_identity()
        functor.forward = MethodType(_identity_default_forward, functor)
        self.assertNotEqual(expected, functor.get_kernel_identity())
        self.assertEqual(float(functor.forward({"input_state": torch.tensor(3.0)}, 0.0, 1.0)["output_state"]), 7.0)

    def test_kernel_identity_detects_in_place_code_replacement(self):
        functor = _CallContractFunctor()
        kernel = type(functor).forward
        expected = functor.get_kernel_identity()
        with _replace_function_attribute(kernel, "__code__", _identity_nested_forward.__code__):
            self.assertNotEqual(expected, functor.get_kernel_identity())
        self.assertEqual(expected, functor.get_kernel_identity())

    def test_kernel_identity_recursively_binds_code_constants(self):
        functor = _CallContractFunctor()
        functor.forward = MethodType(_identity_nested_forward, functor)
        expected = functor.get_kernel_identity()
        original = _identity_nested_forward.__code__
        constants = tuple(
            item.replace(co_consts=tuple(3.0 if constant == 2.0 else constant for constant in item.co_consts))
            if isinstance(item, CodeType) else item
            for item in original.co_consts
        )
        with _replace_function_attribute(_identity_nested_forward, "__code__", original.replace(co_consts=constants)):
            self.assertNotEqual(expected, functor.get_kernel_identity())
            result = functor.forward({"input_state": torch.tensor(3.0)}, 0.0, 1.0)
            self.assertEqual(float(result["output_state"]), 9.0)

    def test_kernel_identity_binds_defaults_and_keyword_defaults(self):
        functor = _CallContractFunctor()
        functor.forward = MethodType(_identity_default_forward, functor)
        expected = functor.get_kernel_identity()
        with _replace_function_attribute(_identity_default_forward, "__defaults__", (4.0,)):
            self.assertNotEqual(expected, functor.get_kernel_identity())
        with _replace_function_attribute(_identity_default_forward, "__kwdefaults__", {"offset": 4.0}):
            self.assertNotEqual(expected, functor.get_kernel_identity())

    def test_kernel_identity_binds_immutable_closure_and_rejects_mutable_cell(self):
        functor = _CallContractFunctor()
        functor.forward = MethodType(_identity_closure_forward(2.0), functor)
        expected = functor.get_kernel_identity()
        functor.forward = MethodType(_identity_closure_forward(2.0), functor)
        self.assertEqual(expected, functor.get_kernel_identity())
        functor.forward = MethodType(_identity_closure_forward(3.0), functor)
        self.assertNotEqual(expected, functor.get_kernel_identity())
        functor.forward = MethodType(_identity_closure_forward([2.0]), functor)
        with self.assertRaisesRegex(ValueError, "closure.*mutable|mutable.*closure"):
            functor.get_kernel_identity()

    def test_kernel_identity_binds_slice_constants(self):
        # Python 3.14 folds constant slices such as x[1::2] into code.co_consts as
        # slice objects; the identity must bind start/stop/step instead of rejecting them.
        functor = _CallContractFunctor()
        functor.forward = MethodType(_identity_default_forward, functor)
        original = _identity_default_forward.__code__
        with_slice = original.replace(co_consts=original.co_consts + (slice(1, None, 2),))
        with _replace_function_attribute(_identity_default_forward, "__code__", with_slice):
            expected = functor.get_kernel_identity()
            self.assertEqual(float(functor.forward({"input_state": torch.tensor(3.0)}, 0.0, 1.0)["output_state"]), 7.0)
        same_slice = original.replace(co_consts=original.co_consts + (slice(1, None, 2),))
        with _replace_function_attribute(_identity_default_forward, "__code__", same_slice):
            self.assertEqual(expected, functor.get_kernel_identity())
        other_step = original.replace(co_consts=original.co_consts + (slice(1, None, 3),))
        with _replace_function_attribute(_identity_default_forward, "__code__", other_step):
            self.assertNotEqual(expected, functor.get_kernel_identity())
        mutable_bound = original.replace(co_consts=original.co_consts + (slice([1], None, None),))
        with _replace_function_attribute(_identity_default_forward, "__code__", mutable_bound):
            with self.assertRaisesRegex(ValueError, "constants\\[.*\\]\\.start .*list"):
                functor.get_kernel_identity()

    def test_kernel_identity_accepts_the_real_class_cell_for_zero_argument_super(self):
        first = _IdentitySuperFunctor()
        second = _IdentitySuperFunctor()
        self.assertEqual(first.get_kernel_identity(), second.get_kernel_identity())
        result = first.forward({"input_state": torch.tensor(3.0)}, 0.0, 1.0)
        self.assertEqual(float(result["output_state"]), 6.0)

    def test_kernel_identity_does_not_unwrap_the_executed_decorator(self):
        functor = _CallContractFunctor()
        functor.forward = MethodType(_identity_default_forward, functor)
        expected = functor.get_kernel_identity()
        functor.forward = MethodType(_identity_wrapped_forward, functor)
        self.assertIs(functor.forward.__wrapped__, _identity_default_forward)
        self.assertNotEqual(expected, functor.get_kernel_identity())
        self.assertEqual(float(functor.forward({"input_state": torch.tensor(3.0)}, 0.0, 1.0)["output_state"]), 5.0)

    def test_kernel_identity_is_independent_of_source_path_and_line_number(self):
        functor = _CallContractFunctor()
        original = _identity_default_forward
        functor.forward = MethodType(original, functor)
        expected = functor.get_kernel_identity()
        with TemporaryDirectory(prefix="relic-x03-identity-") as directory:
            source_path = Path(directory) / "relocated.py"
            source_path.write_text("\n" * 5 + Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
            relocated = FunctionType(
                original.__code__.replace(
                    co_filename=str(source_path),
                    co_firstlineno=original.__code__.co_firstlineno + 5,
                ),
                original.__globals__, original.__name__, original.__defaults__,
                original.__closure__,
            )
            relocated.__module__ = original.__module__
            relocated.__qualname__ = original.__qualname__
            relocated.__kwdefaults__ = original.__kwdefaults__
            functor.forward = MethodType(relocated, functor)
            self.assertEqual(expected, functor.get_kernel_identity())

    def test_kernel_identity_rejects_missing_source_and_opaque_callable(self):
        functor = _CallContractFunctor()
        for opaque in (len, object()):
            with self.subTest(opaque_type=type(opaque).__name__):
                functor.forward = opaque
                with self.assertRaisesRegex(ValueError, "Python"):
                    functor.get_kernel_identity()
        functor.forward = MethodType(_identity_default_forward, functor)
        with _replace_function_attribute(
            _identity_default_forward, "__code__",
            _identity_default_forward.__code__.replace(co_filename="<untraceable-x03>"),
        ), self.assertRaisesRegex(ValueError, "source"):
            functor.get_kernel_identity()

    def test_kernel_identity_rejects_mutable_default_values(self):
        functor = _CallContractFunctor()
        functor.forward = MethodType(_identity_default_forward, functor)
        with _replace_function_attribute(_identity_default_forward, "__defaults__", ([2.0],)), self.assertRaisesRegex(
            ValueError, "default.*mutable|mutable.*default",
        ):
            functor.get_kernel_identity()

    def test_kernel_identity_preserves_immutable_constant_types(self):
        functor = _CallContractFunctor()
        first = (b"\x00\x01", 1.0 + 2.0j, frozenset({"a", "b"}), Ellipsis, -0.0)
        same = (b"\x00\x01", 1.0 + 2.0j, frozenset({"b", "a"}), Ellipsis, -0.0)
        functor.forward = MethodType(_identity_closure_forward(first), functor)
        expected = functor.get_kernel_identity()
        functor.forward = MethodType(_identity_closure_forward(same), functor)
        self.assertEqual(expected, functor.get_kernel_identity())
        functor.forward = MethodType(_identity_closure_forward(first[:-1] + (0.0,)), functor)
        self.assertNotEqual(expected, functor.get_kernel_identity())

    def test_kernel_identity_rejects_foreign_receiver_and_empty_closure(self):
        functor = _CallContractFunctor()
        functor.forward = _CallContractFunctor().forward
        with self.assertRaisesRegex(ValueError, "this Functor instance"):
            functor.get_kernel_identity()
        kernel = _identity_closure_forward(2.0)
        del kernel.__closure__[0].cell_contents
        functor.forward = MethodType(kernel, functor)
        with self.assertRaisesRegex(ValueError, "empty closure"):
            functor.get_kernel_identity()

    def test_kernel_identity_binds_actual_class_with_an_inherited_kernel(self):
        class SameKernelFunctor(_CallContractFunctor):
            pass
        base = _CallContractFunctor().get_kernel_identity()
        derived = SameKernelFunctor().get_kernel_identity()
        self.assertEqual(base["code_sha256"], derived["code_sha256"])
        self.assertNotEqual(base["class_source_sha256"], derived["class_source_sha256"])
        self.assertNotEqual(base, derived)

    def test_kernel_identity_binds_class_helper_source_across_reload(self):
        """Reloaded helper source changes invalidate a shared inherited kernel."""
        module_name = "_relic_x03_class_source_fixture"
        previous = sys.modules.get(module_name)
        with TemporaryDirectory(prefix="relic-x03-class-") as directory:
            source_path = Path(directory) / "source_fixture.py"
            template = "class SourceBoundFunctor(BaseFunctor):\n    def helper(self):\n        return {value}\n"
            source_path.write_text(template.format(value=1), encoding="utf-8")
            spec = importlib.util.spec_from_file_location(module_name, source_path)
            module = importlib.util.module_from_spec(spec)
            module.BaseFunctor = _CallContractFunctor
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                first = module.SourceBoundFunctor()
                first_identity = first.get_kernel_identity()
                self.assertEqual(first.helper(), 1)
                source_path.write_text(template.format(value=1234), encoding="utf-8")
                spec.loader.exec_module(module)
                second = module.SourceBoundFunctor()
                second_identity = second.get_kernel_identity()
                self.assertEqual(second.helper(), 1234)
                self.assertEqual(first_identity["class_qualname"], second_identity["class_qualname"])
                self.assertEqual(first_identity["code_sha256"], second_identity["code_sha256"])
                self.assertNotEqual(first_identity, second_identity)
                type(second).__wrapped__ = _CallContractFunctor
                self.assertEqual(second_identity, second.get_kernel_identity())
            finally:
                if previous is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous

    def test_kernel_identity_rejects_untraceable_class_source(self):
        generated = type("GeneratedKernelFunctor", (_CallContractFunctor,), {})()
        with self.assertRaisesRegex(ValueError, "class source"):
            generated.get_kernel_identity()

    def test_device_management(self):
        """Test device management for tensors."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                # Create tensor on functor's device
                result = torch.zeros(10, device=self.device)
                return {"output": result}

            def get_dependencies(self):
                return []

        # Test with CPU device
        functor = TestFunctor(
            name="test",
            config={},
            metadata=self.test_metadata,
            process_name="test_process",
            device=torch.device("cpu")
        )

        state = {}
        result = functor(state, time=0.0, dt=1.0)
        self.assertEqual(result["output"].device.type, "cpu")

    def test_standard_torch_state_dict_round_trip(self):
        """FunctorBase must preserve the torch.nn.Module load_state_dict API."""
        class TestFunctor(BiologicalProcessFunctor):
            def __init__(self):
                super().__init__(
                    name="state_dict_test",
                    config={},
                    metadata=self_metadata,
                    process_name="state_dict_test_process",
                )
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, state, time, dt, **kwargs):
                return {"output_state": self.weight}

            def get_dependencies(self):
                return []

        self_metadata = self.test_metadata
        functor = TestFunctor()
        restored_state = {"weight": torch.tensor(3.0)}

        result = functor.load_state_dict(restored_state)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(float(functor.weight.detach()), 3.0)

    def test_functor_exposes_no_duplicate_durable_checkpoint_api(self):
        """Durable restart belongs only to the simulator checkpoint owner."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="checkpoint_owner_test",
            config={},
            metadata=self.test_metadata,
            process_name="checkpoint_owner_test_process",
        )

        self.assertFalse(hasattr(functor, "save_checkpoint"))
        self.assertFalse(hasattr(functor, "load_checkpoint"))
        self.assertTrue(callable(functor.state_dict))
        self.assertTrue(callable(functor.load_state_dict))

    def test_validation_checks(self):
        """Test input validation checks."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="test",
            config={},
            metadata=self.test_metadata,
            process_name="test_process"
        )

        # Test with invalid time step
        state = {}
        from relic.core.functor import ExecutionError
        with self.assertRaises(ExecutionError):
            functor(state, time=0.0, dt=-1.0)  # Negative dt

    def test_dependency_declaration(self):
        """Test dependency declaration system."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {}

            def get_dependencies(self):
                return ["dependency_1", "dependency_2"]

        functor = TestFunctor(
            name="test",
            config={},
            metadata=self.test_metadata,
            process_name="test_process"
        )

        deps = functor.get_dependencies()
        self.assertEqual(len(deps), 2)
        self.assertIn("dependency_1", deps)
        self.assertIn("dependency_2", deps)

    def test_flux_dict_structure(self):
        """Test that forward returns proper FluxDict structure."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                return {
                    "carbon_flux": torch.tensor(1.0),
                    "nitrogen_flux": torch.tensor(0.5),
                    "diagnostic_value": torch.tensor(2.0)
                }

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="test",
            config={},
            metadata=self.test_metadata,
            process_name="test_process"
        )

        state = {}
        result = functor(state, time=0.0, dt=1.0)

        self.assertIsInstance(result, dict)
        self.assertIn("carbon_flux", result)
        self.assertIsInstance(result["carbon_flux"], torch.Tensor)

    def test_error_handling(self):
        """Test error handling and exception propagation."""
        class TestFunctor(BiologicalProcessFunctor):
            def forward(self, state, time, dt, **kwargs):
                raise RuntimeError("Test error")

            def get_dependencies(self):
                return []

        functor = TestFunctor(
            name="test",
            config={},
            metadata=self.test_metadata,
            process_name="test_process"
        )

        state = {}
        from relic.core.functor import ExecutionError
        with self.assertRaises(ExecutionError):
            functor(state, time=0.0, dt=1.0)


if __name__ == "__main__":
    unittest.main()



class _BindingManager:
    """A ParameterManager stand-in holding learnable bindings in nn_parameters."""

    def __init__(self, **values):
        self.nn_parameters = {
            key.replace("__", "."): torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for key, value in values.items()
        }

    def get_parameter(self, key):
        """No declared specification: the stand-in holds bindings only."""
        return None

    def get(self, key, **kwargs):
        """The bridge's declared pH optimum; every other value is absent."""
        return {"validation.numerical.response.ph_optimum": 7.0}.get(key)


def _holder_metadata(name):
    return FunctorMetadata(
        name=name,
        category=ProcessCategory.BIOLOGICAL_PROCESS,
        execution_phase=ExecutionPhase.MAIN_PROCESSES,
        version="1.0.0",
        description="Functor holding a ParameterManager binding",
        references=["Unit test reference"],
        input_variables=["x"],
        output_variables=["output_state"],
    )


class _HoldingFunctor(BiologicalProcessFunctor):
    """Reads theta from its manager binding and owns one private Parameter."""

    def __init__(self, manager, name="holder"):
        super().__init__(
            name=name,
            config={},
            metadata=_holder_metadata(name),
            parameter_manager=manager,
        )
        self.theta = manager.nn_parameters["test.theta"]
        self.own = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def forward(self, state, time, dt, **kwargs):
        return {"output_state": self.theta * self.own * state["x"]}

    def get_dependencies(self):
        return []


class TestManagerHeldParametersL1(unittest.TestCase):
    """A learnable ParameterManager binding has one owner, the manager.

    References: docs/source/architecture_overview.md Section 4.2 (parameters are
    owned by ParameterManager). A Functor attribute that is, by object identity,
    a binding of the Functor's own manager is held by reference: it is not a
    module parameter, so parameters(), state_dict(), .to() and .double() of the
    Functor never copy, save or convert it, and every holder reads one theta.
    """

    def setUp(self):
        self.manager = _BindingManager(test__theta=2.0)
        self.binding = self.manager.nn_parameters["test.theta"]

    def test_binding_is_held_by_reference_not_registered(self):
        functor = _HoldingFunctor(self.manager)
        self.assertIs(functor.theta, self.binding)
        self.assertNotIn("theta", dict(functor.named_parameters()))
        self.assertNotIn("theta", functor.state_dict())
        self.assertIn("own", dict(functor.named_parameters()))
        self.assertEqual(list(functor.held_manager_parameters()), ["test.theta"])
        self.assertIs(functor.held_manager_parameters()["test.theta"], self.binding)
        self.assertEqual(functor.manager_binding_key(self.binding), "test.theta")
        self.assertIsNone(functor.manager_binding_key(functor.own))

    def test_mapping_held_binding_keeps_the_original_manager_identity(self):
        functor = _HoldingFunctor(self.manager)
        del functor.theta
        other = _BindingManager(test__theta=2.0)
        private = torch.nn.Parameter(self.binding.detach().clone())
        view = self.binding.view_as(self.binding)
        functor.process_parameters = {
            "theta": self.binding,
            "foreign": other.nn_parameters["test.theta"],
            "private": private,
            "view": view,
        }

        held = functor.held_manager_parameters()
        self.assertEqual(list(held), ["test.theta"])
        self.assertIs(held["test.theta"], self.binding)
        self.assertIs(functor.process_parameters["theta"], self.binding)
        self.assertEqual(list(functor.named_parameters()), [("own", functor.own)])
        self.assertNotIn("process_parameters.theta", functor.state_dict())
        # References: PyTorch autograd; architecture SSOT Section 4.2.
        # Mathematical expression: L(theta)=theta^2; dL/dtheta=2*theta.
        # Parameter meanings: theta is the original manager-owned test binding.
        # Units: theta, L and its derivative [1] in this manufactured control.
        # Domain/constraints: theta=2; no copies or production process calls.
        # Sign convention: positive theta gives a positive derivative.
        functor.process_parameters["theta"].square().backward()
        self.assertEqual(float(self.binding.grad), 4.0)
        self.assertIsNone(private.grad)
        self.assertIsNone(other.nn_parameters["test.theta"].grad)

    def test_nested_container_aliases_and_cycles_find_one_binding(self):
        functor = _HoldingFunctor(self.manager)
        del functor.theta
        containers = {
            "mapping": MappingProxyType({"theta": self.binding}),
            "mapping_key": {self.binding: "binding_key"},
            "list": [self.binding],
            "tuple": (self.binding,),
            "set": {self.binding},
            "frozenset": frozenset({self.binding}),
        }
        for name, container in containers.items():
            with self.subTest(container=name):
                functor.process_parameters = container
                held = functor.held_manager_parameters()
                self.assertEqual(list(held), ["test.theta"])
                self.assertIs(held["test.theta"], self.binding)
        shared = {"theta": self.binding}
        cyclic = [MappingProxyType(shared), (shared,), {self.binding}, frozenset({self.binding})]
        cyclic.append(cyclic)
        shared["cycle"] = cyclic
        functor.process_parameters = {"nested": cyclic}

        held = functor.held_manager_parameters()
        self.assertEqual(list(held), ["test.theta"])
        self.assertIs(held["test.theta"], self.binding)
        self.assertIs(shared["cycle"], cyclic)
        self.assertIs(cyclic[-1], cyclic)
        self.assertEqual(list(functor.named_parameters()), [("own", functor.own)])

    def test_container_discovery_tracks_only_current_holders(self):
        functor = _HoldingFunctor(self.manager)
        del functor.theta
        functor.process_parameters = {"theta": self.binding}
        self.assertIs(functor.held_manager_parameters()["test.theta"], self.binding)
        functor.process_parameters.clear()
        # A manager object and a nested Functor remain separate lifecycle owners.
        functor.owner_reference = self.manager
        functor.nested = _HoldingFunctor(self.manager, name="nested_holder")
        self.assertEqual(functor.held_manager_parameters(), {})
        self.assertIs(functor.nested.held_manager_parameters()["test.theta"], self.binding)
        functor.process_parameters = {"theta": self.binding}
        held = functor.held_manager_parameters()
        held.clear()
        self.assertIs(functor.held_manager_parameters()["test.theta"], self.binding)

    def test_module_conversions_leave_the_binding_untouched(self):
        functor = _HoldingFunctor(self.manager)
        functor.double()
        self.assertIs(functor.theta, self.binding)
        self.assertEqual(self.binding.dtype, torch.float32)
        self.assertEqual(functor.own.dtype, torch.float64)
        if torch.cuda.is_available():
            functor.to_device("cuda")
            self.assertEqual(self.binding.device.type, "cpu")
            later = _HoldingFunctor(self.manager, name="later_holder")
            self.assertIs(later.theta, self.binding)

    def test_identity_not_name_or_value_decides(self):
        functor = _HoldingFunctor(self.manager)
        # An equal-valued Parameter that is not the binding is the Functor's own.
        functor.copy_theta = torch.nn.Parameter(self.binding.detach().clone())
        self.assertIn("copy_theta", dict(functor.named_parameters()))
        # A binding of another manager is no binding of this Functor's manager.
        other = _BindingManager(test__theta=2.0)
        functor.foreign = other.nn_parameters["test.theta"]
        self.assertIn("foreign", dict(functor.named_parameters()))
        self.assertEqual(list(functor.held_manager_parameters()), ["test.theta"])

    def test_reassignment_moves_between_registration_and_reference(self):
        functor = _HoldingFunctor(self.manager)
        functor.own = self.binding
        self.assertNotIn("own", dict(functor.named_parameters()))
        self.assertIs(functor.own, self.binding)
        functor.theta = torch.nn.Parameter(torch.tensor(4.0))
        self.assertIn("theta", dict(functor.named_parameters()))
        del functor.own
        self.assertEqual(functor.held_manager_parameters(), {})

    def test_gradients_of_two_holders_reach_the_one_binding(self):
        first = _HoldingFunctor(self.manager, name="first_holder")
        second = _HoldingFunctor(self.manager, name="second_holder")
        state = {"x": torch.tensor(3.0)}
        total = (
            first(state, time=0.0, dt=1.0)["output_state"]
            + second(state, time=0.0, dt=1.0)["output_state"]
        )
        total.backward()
        # d/dtheta [theta * own * x] summed over both holders = 2 * 1.5 * 3.
        self.assertAlmostEqual(float(self.binding.grad), 9.0, places=6)

    def test_re_registering_a_held_key_keeps_the_binding_object(self):
        """ParameterManager.register_parameter of an existing learnable key (as a
        database reload does) keeps a binding of the same shape and dtype: the
        holders keep reading, and the manager keeps optimizing and checkpointing,
        the one theta. A new binding is made only when shape or dtype change."""
        from dataclasses import replace

        from relic.parameters import ParameterSpec, UnifiedParameterManager

        manager = UnifiedParameterManager()
        spec = ParameterSpec(
            key="test.theta", value=2.0, units="dimensionless", dtype="float32",
            bounds=(0.0, 10.0), category="calibration_parameter", requires_grad=True,
        )
        manager.register_parameter(spec)
        functor = _HoldingFunctor(manager)
        binding = manager.nn_parameters["test.theta"]
        self.assertIs(functor.theta, binding)
        binding.requires_grad_(False)

        manager.register_parameter(replace(spec, value=3.0))
        self.assertIs(manager.nn_parameters["test.theta"], binding)
        self.assertIs(manager.get_nn_parameters()["test.theta"], functor.theta)
        self.assertEqual(functor.held_manager_parameters(), {"test.theta": binding})
        self.assertEqual(float(functor.theta), 3.0)
        self.assertFalse(binding.requires_grad)

        manager.register_parameter(replace(spec, value=3.0, dtype="float64"))
        self.assertIsNot(manager.nn_parameters["test.theta"], binding)
        self.assertEqual(manager.nn_parameters["test.theta"].dtype, torch.float64)
        self.assertEqual(functor.held_manager_parameters(), {})

    def test_no_duplicate_parameter_listing_api(self):
        from relic.core.functor import FunctorBase

        self.assertFalse(hasattr(FunctorBase, "get_torch_parameters"))


class _SignedTemperatureFunctor(BiologicalProcessFunctor):
    """Fixture declaring an absolute temperature and a signed anomaly input."""

    def __init__(self):
        super().__init__(
            name="signed_temperature",
            config={},
            metadata=FunctorMetadata(
                name="signed_temperature",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                description="Temperature sign heuristic fixture",
                references=["Hasselmann (1976), Tellus 28, 473-485"],
                input_variables=["air_temperature", "air_temperature_anomaly"],
                output_variables=["output_state"],
            ),
            process_name="signed_temperature",
        )

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"output_state": state["air_temperature"] + state["air_temperature_anomaly"]}


class TemperatureSignHeuristicTests(unittest.TestCase):
    """Absolute kelvin temperatures stay non-negative; anomalies are signed."""

    def test_negative_anomaly_is_accepted_but_negative_kelvin_is_not(self):
        functor = _SignedTemperatureFunctor()
        accepted = functor.validate_state({
            "air_temperature": torch.tensor(280.0),
            "air_temperature_anomaly": torch.tensor(-3.5),
        })
        self.assertEqual(accepted, [])
        rejected = functor.validate_state({
            "air_temperature": torch.tensor(-1.0),
            "air_temperature_anomaly": torch.tensor(-3.5),
        })
        self.assertEqual(len(rejected), 1)
        self.assertIn("air_temperature", rejected[0])
        self.assertNotIn("anomaly", rejected[0])


class _OptionalInputRuleFunctor(BiologicalProcessFunctor):
    """Declare three inputs; the constructor names the optional ones."""

    def __init__(self, optional_inputs=("wind_speed",)):
        super().__init__(
            name="optional_input_rule",
            config={},
            metadata=FunctorMetadata(
                name="optional_input_rule",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                description="Required-input rule fixture",
                references=["RELIC FunctorBase required-input contract"],
                input_variables=["air_temperature", "wind_speed", "air_pressure"],
                output_variables=["output_state"],
            ),
            process_name="optional_input_rule",
        )
        self.optional_input_variables = set(optional_inputs)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"output_state": state["air_temperature"]}


class TestRequiredInputRuleL1(unittest.TestCase):
    """One required-input rule: declared inputs minus optional_input_variables."""

    def test_required_inputs_keep_declaration_order_without_optional_inputs(self):
        self.assertEqual(
            _OptionalInputRuleFunctor().get_required_inputs(),
            ["air_temperature", "air_pressure"],
        )
        self.assertEqual(
            _OptionalInputRuleFunctor(optional_inputs=()).get_required_inputs(),
            ["air_temperature", "wind_speed", "air_pressure"],
        )

    def test_validate_state_checks_only_the_required_inputs(self):
        errors = _OptionalInputRuleFunctor().validate_state({"wind_speed": torch.tensor(2.0)})
        self.assertEqual(
            errors,
            [
                "Required state variable 'air_temperature' not found",
                "Required state variable 'air_pressure' not found",
            ],
        )

    def test_node_registry_reads_requiredness_from_the_functor_rule(self):
        from relic.orchestration.node_registry import NodeRegistry
        from relic.orchestration.ssot_catalog import SSOTCatalog

        definition = NodeRegistry(SSOTCatalog()).register_functor_instance(
            _OptionalInputRuleFunctor()
        )
        self.assertEqual(
            {name: port.required for name, port in definition.inputs.items()},
            {"air_temperature": True, "wind_speed": False, "air_pressure": True},
        )

    def test_node_registry_rejects_an_instance_without_the_rule(self):
        from types import SimpleNamespace

        from relic.orchestration.node_registry import NodeRegistry
        from relic.orchestration.ssot_catalog import SSOTCatalog

        bare = SimpleNamespace(
            metadata=SimpleNamespace(
                input_variables=["air_temperature"],
                output_variables=["output_state"],
                to_dict=lambda: {"name": "bare"},
            )
        )
        with self.assertRaisesRegex(TypeError, "get_required_inputs"):
            NodeRegistry(SSOTCatalog()).register_functor_instance(bare)

    def test_no_functor_class_overrides_the_required_input_rule(self):
        """optional_input_variables is the only declaration of optional inputs."""
        import ast

        relic_root = Path(__file__).resolve().parents[2] / "relic"
        allowed = {
            ("core/functor/functor_base.py", "FunctorBase"),
            ("scientific/pedosphere/api_contract.py", "PedosphereFunctorContract"),
        }
        overrides = []
        for path in sorted(relic_root.rglob("*.py")):
            relative = path.relative_to(relic_root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "get_required_inputs"
                        and (relative, node.name) not in allowed
                    ):
                        overrides.append(f"{relative}:{node.name}")
        self.assertEqual(overrides, [])
