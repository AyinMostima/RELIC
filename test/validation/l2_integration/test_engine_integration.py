"""
L2 Integration Tests: Validation Engine Integration

Tests for validation engine orchestration, rule dependency resolution,
caller-thread execution, and comprehensive reporting integration.

Focus Areas:
1. Engine Rule Orchestration and Execution
2. Dependency Resolution and Topological Sorting
3. Sequential Rule Execution and External Caller Identity
4. Report Generation and Aggregation
5. Large Rule-Set Functional Coverage and Observed Timing

Mathematical Framework:
    Dependency resolution using topological sorting:
    - Graph G = (V, E) where V = validation rules, E = dependencies
    - Topological order ensures dependencies execute before dependents
    - Independent rules still execute sequentially in the calling thread
    
References:
    - Cormen et al. (2009) Introduction to Algorithms - Topological Sort
    - Validation framework architecture patterns
"""

import math
import pytest
import time
import threading

from relic.validation.framework.models import (
    ValidationLevel, ValidationCode, ValidationResultModel, ValidationReportModel, ValidationContext
)
from relic.validation.framework.registry import ValidationRegistry
from relic.validation.framework.engine import ValidationEngine
from relic.validation.framework.decorators import validated_rule


class TestEngineRuleOrchestration:
    """Test validation engine rule orchestration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
        self.registry = ValidationRegistry()
    
    
    def test_single_rule_execution(self):
        """Test execution of single validation rule."""
        @validated_rule(
            rule_id="test_single_rule",
            layer="L2",
            domains=["test"],
            tags=["integration"]
        )
        def test_single_rule(context):
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Single rule test passed",
                layer="L2",
                scope="test_single_rule"
            )]
        
        # Execute single rule
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2"
        )
        
        report = self.engine.run(layers="L2", context=context)
        
        assert isinstance(report, ValidationReportModel)
        assert len(report.results) >= 1
        
        # Find our test rule result
        test_results = [r for r in report.results if r.scope == "test_single_rule"]
        assert len(test_results) == 1
        assert test_results[0].message == "Single rule test passed"
        assert test_results[0].layer == "L2"
    
    def test_multiple_rule_execution(self):
        """Test execution of multiple validation rules."""
        @validated_rule(
            rule_id="test_rule_a",
            layer="L2",
            domains=["test"],
            tags=["integration"]
        )
        def test_rule_a(context):
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Rule A executed",
                layer="L2",
                scope="test_rule_a"
            )]
        
        @validated_rule(
            rule_id="test_rule_b", 
            layer="L2",
            domains=["test"],
            tags=["integration"]
        )
        def test_rule_b(context):
            return [ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.CF_UNIT_MISMATCH,
                message="Rule B executed with warning",
                layer="L2",
                scope="test_rule_b"
            )]
        
        @validated_rule(
            rule_id="test_rule_c",
            layer="L2", 
            domains=["test"],
            tags=["integration"]
        )
        def test_rule_c(context):
            return [ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
                message="Rule C executed with error",
                layer="L2",
                scope="test_rule_c"
            )]
        
        # Execute multiple rules
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2"
        )
        
        report = self.engine.run(layers="L2", context=context)
        
        # Verify all rules executed
        test_scopes = [r.scope for r in report.results if r.scope.startswith("test_rule_")]
        assert "test_rule_a" in test_scopes
        assert "test_rule_b" in test_scopes
        assert "test_rule_c" in test_scopes
        
        # Verify summary statistics
        assert report.summary["total"] >= 3
        assert report.summary["by_level"]["INFO"] >= 1
        assert report.summary["by_level"]["WARNING"] >= 1
        assert report.summary["by_level"]["ERROR"] >= 1
        
        # Verify status methods
        assert report.has_warnings() is True
        assert report.has_errors() is True
    
    def test_domain_filtering(self):
        """Test domain-based rule filtering."""
        @validated_rule(
            rule_id="atmosphere_rule",
            layer="L2",
            domains=["atmosphere"],
            tags=["domain_test"]
        )
        def atmosphere_rule(context):
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Atmosphere rule executed",
                layer="L2",
                scope="atmosphere_rule"
            )]
        
        @validated_rule(
            rule_id="biosphere_rule",
            layer="L2",
            domains=["biosphere"],
            tags=["domain_test"]
        )
        def biosphere_rule(context):
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Biosphere rule executed",
                layer="L2", 
                scope="biosphere_rule"
            )]
        
        # Test atmosphere domain only
        context_atm = ValidationContext(
            domain_filter=["atmosphere"],
            execution_layer="L2"
        )
        
        report_atm = self.engine.run(layers="L2", domains=["atmosphere"], tags=["domain_test"], context=context_atm)
        atm_scopes = [r.scope for r in report_atm.results]
        
        assert atm_scopes == ["atmosphere_rule"]
        
        # Test biosphere domain only  
        context_bio = ValidationContext(
            domain_filter=["biosphere"],
            execution_layer="L2"
        )
        
        report_bio = self.engine.run(layers="L2", domains=["biosphere"], tags=["domain_test"], context=context_bio)
        bio_scopes = [r.scope for r in report_bio.results]
        
        assert bio_scopes == ["biosphere_rule"]


class TestDependencyResolution:
    """Test validation rule dependency resolution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
        self.registry = ValidationRegistry()
    
    
    def test_simple_dependency_chain(self):
        """Test simple dependency chain execution."""
        execution_order = []
        
        @validated_rule(
            rule_id="base_rule",
            layer="L2",
            domains=["test"],
            dependencies=[],
            tags=["dependency_test"]
        )
        def base_rule(context):
            execution_order.append("base_rule")
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Base rule executed",
                layer="L2",
                scope="base_rule"
            )]
        
        @validated_rule(
            rule_id="dependent_rule",
            layer="L2",
            domains=["test"],
            dependencies=["base_rule"],
            tags=["dependency_test"]
        )
        def dependent_rule(context):
            execution_order.append("dependent_rule")
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Dependent rule executed",
                layer="L2",
                scope="dependent_rule"
            )]
        
        # Execute with dependencies
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2"
        )
        
        report = self.engine.run(layers="L2", context=context)
        
        # Verify execution order
        assert len(execution_order) >= 2
        base_idx = execution_order.index("base_rule")
        dependent_idx = execution_order.index("dependent_rule")
        assert base_idx < dependent_idx  # Base rule must execute before dependent
        
        # Verify both rules executed successfully
        test_scopes = [r.scope for r in report.results if r.scope in ["base_rule", "dependent_rule"]]
        assert "base_rule" in test_scopes
        assert "dependent_rule" in test_scopes
    
    def test_complex_dependency_graph(self):
        """Test complex dependency graph resolution."""
        execution_order = []
        
        @validated_rule(
            rule_id="root_rule",
            layer="L2",
            domains=["test"],
            dependencies=[],
            tags=["complex_dependency"]
        )
        def root_rule(context):
            execution_order.append("root_rule")
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Root rule executed",
                layer="L2",
                scope="root_rule"
            )]
        
        @validated_rule(
            rule_id="branch_a",
            layer="L2",
            domains=["test"],
            dependencies=["root_rule"],
            tags=["complex_dependency"]
        )
        def branch_a(context):
            execution_order.append("branch_a")
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Branch A executed",
                layer="L2",
                scope="branch_a"
            )]
        
        @validated_rule(
            rule_id="branch_b",
            layer="L2",
            domains=["test"],
            dependencies=["root_rule"],
            tags=["complex_dependency"]
        )
        def branch_b(context):
            execution_order.append("branch_b")
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Branch B executed",
                layer="L2",
                scope="branch_b"
            )]
        
        @validated_rule(
            rule_id="merge_rule",
            layer="L2",
            domains=["test"],
            dependencies=["branch_a", "branch_b"],
            tags=["complex_dependency"]
        )
        def merge_rule(context):
            execution_order.append("merge_rule")
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Merge rule executed",
                layer="L2",
                scope="merge_rule"
            )]
        
        # Execute complex dependency graph
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2"
        )
        
        report = self.engine.run(layers="L2", context=context)
        
        # Verify topological ordering
        assert len(execution_order) >= 4
        
        root_idx = execution_order.index("root_rule")
        branch_a_idx = execution_order.index("branch_a")
        branch_b_idx = execution_order.index("branch_b")
        merge_idx = execution_order.index("merge_rule")
        
        # Root must execute before branches
        assert root_idx < branch_a_idx
        assert root_idx < branch_b_idx
        
        # Branches must execute before merge
        assert branch_a_idx < merge_idx
        assert branch_b_idx < merge_idx
        
        # All rules should have executed successfully
        expected_scopes = {"root_rule", "branch_a", "branch_b", "merge_rule"}
        actual_scopes = {r.scope for r in report.results if r.scope in expected_scopes}
        assert expected_scopes.issubset(actual_scopes)
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # Register circular dependency rules - they get registered successfully
        @validated_rule(
            rule_id="circular_rule_a",
            layer="L2",
            domains=["test"],
            dependencies=["circular_rule_b"],
            tags=["circular_test"]
        )
        def circular_rule_a(context):
            return []
        
        @validated_rule(
            rule_id="circular_rule_b",
            layer="L2",
            domains=["test"],
            dependencies=["circular_rule_a"],  # Creates circular dependency
            tags=["circular_test"]
        )
        def circular_rule_b(context):
            return []
        
        # Execution should detect the circular dependency and handle it gracefully
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2"
        )
        
        # The engine should detect the circular dependency and produce error result
        report = self.engine.run(layers="L2", context=context)
        
        # Check that the engine handled the circular dependency gracefully
        assert isinstance(report, ValidationReportModel)
        # The circular dependency should be handled by producing an error result
        # rather than crashing the entire execution
        circular_errors = [r for r in report.results 
                         if "Circular dependency" in r.message or r.level == ValidationLevel.CRITICAL]
        assert len(circular_errors) >= 1  # Should have detected and reported the circular dependency
        
        # Clean up the problematic rules after the test
        try:
            self.engine.registry.unregister_rule("circular_rule_a")
            self.engine.registry.unregister_rule("circular_rule_b")
        except:
            pass  # Ignore cleanup errors


class TestCallerThreadExecution:
    """Rules execute sequentially within each external caller; no engine worker pool is claimed."""

    def setup_method(self):
        self.engine = ValidationEngine()

    def test_independent_rules_execute_once_in_the_calling_thread(self):
        caller_thread = threading.get_ident()
        events = []

        @validated_rule(
            rule_id="caller_rule_1", layer="L2", domains=["test"],
            dependencies=[], tags=["caller_order_contract"],
        )
        def caller_rule_1(context):
            events.append(("start", "caller_rule_1", threading.get_ident()))
            result = ValidationResultModel(
                level=ValidationLevel.INFO, code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Caller rule 1 executed", layer="L2", scope="caller_rule_1",
            )
            events.append(("end", "caller_rule_1", threading.get_ident()))
            return [result]

        @validated_rule(
            rule_id="caller_rule_2", layer="L2", domains=["test"],
            dependencies=[], tags=["caller_order_contract"],
        )
        def caller_rule_2(context):
            events.append(("start", "caller_rule_2", threading.get_ident()))
            result = ValidationResultModel(
                level=ValidationLevel.INFO, code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Caller rule 2 executed", layer="L2", scope="caller_rule_2",
            )
            events.append(("end", "caller_rule_2", threading.get_ident()))
            return [result]

        context = ValidationContext(domain_filter=["test"], execution_layer="L2")
        report = self.engine.run(
            layers="L2", domains=["test"], tags=["caller_order_contract"], context=context,
        )
        assert len(events) == 4
        names = [events[0][1], events[2][1]]
        assert set(names) == {"caller_rule_1", "caller_rule_2"}
        assert events == [(phase, name, caller_thread) for name in names for phase in ("start", "end")]
        assert [result.scope for result in report.results] == names
        assert all(result.level == ValidationLevel.INFO for result in report.results)
        assert report.summary["total"] == 2
        assert not report.has_errors()

    def test_external_callers_keep_rule_execution_in_their_own_threads(self):
        # Only external-call identity and independent results are claimed, not arbitrary shared-state safety.
        execution_records = []
        caller_threads = {}
        reports = {}
        errors = []
        records_lock = threading.Lock()

        @validated_rule(
            rule_id="caller_identity_rule", layer="L2", domains=["test"],
            tags=["external_caller_contract"],
        )
        def caller_identity_rule(context):
            caller_index = context.metadata["caller_index"]
            thread_id = threading.get_ident()
            with records_lock:
                execution_records.append((caller_index, thread_id))
            return [ValidationResultModel(
                level=ValidationLevel.INFO, code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message=f"Caller {caller_index} executed on thread {thread_id}",
                layer="L2", scope="caller_identity_rule",
                metrics={"caller_index": caller_index, "thread_id": thread_id},
            )]

        def run_validation(caller_index):
            thread_id = threading.get_ident()
            context = ValidationContext(
                domain_filter=["test"], execution_layer="L2", metadata={"caller_index": caller_index},
            )
            try:
                report = self.engine.run(
                    layers="L2", domains=["test"], tags=["external_caller_contract"], context=context,
                )
                with records_lock:
                    caller_threads[caller_index] = thread_id
                    reports[caller_index] = report
            except Exception as exc:
                with records_lock:
                    errors.append((caller_index, type(exc).__name__, str(exc)))

        threads = [threading.Thread(target=run_validation, args=(index,)) for index in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        assert set(reports) == set(caller_threads) == {0, 1, 2}
        assert sorted(execution_records) == sorted(caller_threads.items())
        assert all(thread_id != threading.get_ident() for thread_id in caller_threads.values())
        for caller_index, report in reports.items():
            assert isinstance(report, ValidationReportModel)
            assert [result.scope for result in report.results] == ["caller_identity_rule"]
            assert report.results[0].metrics == {
                "caller_index": caller_index, "thread_id": caller_threads[caller_index],
            }
            assert report.summary["total"] == 1
            assert not report.has_errors()


class TestReportGeneration:
    """Test comprehensive report generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    
    def test_comprehensive_report_generation(self):
        """Test generation of comprehensive validation reports."""
        @validated_rule(
            rule_id="info_rule",
            layer="L2",
            domains=["test"],
            tags=["report_test", "info"]
        )
        def info_rule(context):
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Information level validation",
                layer="L2",
                scope="info_rule",
                tags=["info", "test"]
            )]
        
        @validated_rule(
            rule_id="warning_rule",
            layer="L2",
            domains=["test"],
            tags=["report_test", "warning"]
        )
        def warning_rule(context):
            return [ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.CF_UNIT_MISMATCH,
                message="Warning level validation",
                layer="L2",
                scope="warning_rule",
                tags=["warning", "cf"]
            )]
        
        @validated_rule(
            rule_id="error_rule",
            layer="L2",
            domains=["test"],
            tags=["report_test", "error"]
        )
        def error_rule(context):
            return [ValidationResultModel(
                level=ValidationLevel.ERROR,
                code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
                message="Error level validation",
                layer="L2",
                scope="error_rule",
                tags=["error", "conservation"]
            )]
        
        # Execute comprehensive validation
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2",
            metadata={"test_run": True, "report_test": True}
        )
        
        report = self.engine.run(layers="L2", context=context)
        
        # Verify report structure
        assert isinstance(report, ValidationReportModel)
        assert hasattr(report, 'results')
        assert hasattr(report, 'summary')
        assert hasattr(report, 'cf_summary')
        assert hasattr(report, 'conservation_summary')
        assert hasattr(report, 'execution_metadata')
        
        # Verify summary statistics
        assert report.summary["total"] >= 3
        assert report.summary["by_level"]["INFO"] >= 1
        assert report.summary["by_level"]["WARNING"] >= 1
        assert report.summary["by_level"]["ERROR"] >= 1
        assert report.summary["by_layer"]["L2"] >= 3
        
        # Verify domain-specific summaries
        assert report.cf_summary["total_cf_validations"] >= 1
        assert report.cf_summary["cf_warnings"] >= 1
        
        assert report.conservation_summary["total_conservation_checks"] >= 1
        assert report.conservation_summary["conservation_failures"] >= 1
        
        # Test report filtering methods
        info_results = report.get_results_by_level(ValidationLevel.INFO)
        warning_results = report.get_results_by_level(ValidationLevel.WARNING)
        error_results = report.get_results_by_level(ValidationLevel.ERROR)
        
        assert len(info_results) >= 1
        assert len(warning_results) >= 1
        assert len(error_results) >= 1
        
        # Test filtering by layer
        l2_results = report.get_results_by_layer("L2")
        assert len(l2_results) >= 3
        
        # Test filtering by tags
        cf_results = report.get_results_by_tag("cf")
        conservation_results = report.get_results_by_tag("conservation")
        
        assert len(cf_results) >= 1
        assert len(conservation_results) >= 1
        
        # Test status methods
        assert report.has_warnings() is True
        assert report.has_errors() is True
    
    def test_report_serialization(self):
        """Test report serialization to different formats."""
        @validated_rule(
            rule_id="serialization_test_rule",
            layer="L2",
            domains=["test"],
            tags=["serialization"]
        )
        def serialization_test_rule(context):
            result = ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.CF_UNIT_MISMATCH,
                message="Serialization test validation",
                layer="L2",
                scope="serialization_test_rule"
            )
            
            # Add detail for richer serialization
            result.add_detail(
                variable_name="test_variable",
                cf_standard_name="test_standard_name",
                units="test_units",
                value=42.0,
                threshold=50.0,
                statistics={"mean": 40.0, "std": 5.0}
            )
            
            return [result]
        
        # Execute validation
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2"
        )
        
        report = self.engine.run(layers="L2", context=context)
        
        # Test dictionary serialization
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "results" in report_dict
        assert "summary" in report_dict
        assert "timestamp" in report_dict
        
        # Verify results structure
        assert len(report_dict["results"]) >= 1
        test_result = next((r for r in report_dict["results"] 
                          if r["scope"] == "serialization_test_rule"), None)
        assert test_result is not None
        assert test_result["level"] == "WARNING"
        assert test_result["message"] == "Serialization test validation"
        
        # Verify details are serialized
        if test_result.get("details"):
            detail = test_result["details"][0]
            assert detail["variable_name"] == "test_variable"
            assert detail["value"] == 42.0
            assert detail["statistics"]["mean"] == 40.0


class TestEngineRuleSetCoverage:
    """Test complete execution of a larger rule set, with timing as observation only."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    
    def test_large_rule_set_executes_each_rule_once(self):
        """Keep all 50 rules and verify their results without a wall-clock win threshold."""
        # Register multiple rules programmatically
        num_rules = 50
        execution_counts = {index: 0 for index in range(num_rules)}
        execution_threads = []
        caller_thread = threading.get_ident()
        
        for i in range(num_rules):
            rule_id = f"performance_rule_{i}"
            
            @validated_rule(
                rule_id=rule_id,
                layer="L2",
                domains=["test"],
                tags=["large_rule_set_functional", f"batch_{i // 10}"]
            )
            def performance_rule(context, rule_num=i):
                execution_counts[rule_num] += 1
                execution_threads.append(threading.get_ident())
                
                return [ValidationResultModel(
                    level=ValidationLevel.INFO,
                    code=ValidationCode.GENERIC_VALIDATION_FAILED,
                    message=f"Performance rule {rule_num} executed",
                    layer="L2",
                    scope=f"performance_rule_{rule_num}"
                )]
        
        # Observe elapsed time without using it as a correctness or scientific-efficiency gate.
        start_time = time.perf_counter()
        
        context = ValidationContext(
            domain_filter=["test"],
            execution_layer="L2"
        )
        
        report = self.engine.run(
            layers="L2", domains=["test"], tags=["large_rule_set_functional"], context=context,
        )
        execution_time = time.perf_counter() - start_time
        assert all(count == 1 for count in execution_counts.values())
        assert execution_threads == [caller_thread] * num_rules
        assert len(report.results) == num_rules
        assert {result.scope for result in report.results} == {f"performance_rule_{i}" for i in range(num_rules)}
        assert report.summary["total"] == num_rules
        assert report.summary["by_layer"]["L2"] == num_rules
        assert not report.has_errors()
        assert math.isfinite(execution_time) and execution_time >= 0.0
        print(f"Rule-set observation: rules={num_rules}, elapsed_seconds={execution_time:.9f}, caller_thread={caller_thread}")


if __name__ == "__main__":
    pytest.main([__file__])

