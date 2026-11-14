"""
L1 Unit Tests - FunctorBase

Tests the unified functor base class which provides the foundation for all
scientific process modules with dependency management, parameter handling,
and conservation checking.
"""

import pytest
import torch
from typing import Dict, List

# Import functor base and metadata classes
from relic.core.functor import (
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    FunctorState,
    ExecutionError,
)
import relic.validation.framework.functor_integration as functor_integration
from relic.validation.framework.functor_integration import (
    FunctorMetadataIntegrator,
    get_global_integrator,
    register_functor_validations,
)
from relic.validation.framework.models import ValidationContext
from relic.validation.framework.registry import (
    ValidationRegistry,
    get_global_registry,
    reset_global_registry,
)


class TestFunctorMetadata:
    """Test suite for FunctorMetadata."""
    
    def test_metadata_creation(self):
        """Test creation of functor metadata."""
        metadata = FunctorMetadata(
            name="TestFunctor",
            version="1.0.0",
            description="Test functor for unit testing",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="dX/dt = -k*X",
            references=["Test et al. (2024)"],
            parameters={
                "k": "Decay rate constant [1/s]"
            },
            input_variables=["X"],
            output_variables=["dX_dt"],
            conservation_laws=["mass"],
            spatial_dimensions=2,
            temporal_scale="second"
        )
        
        assert metadata.name == "TestFunctor"
        assert metadata.category == ProcessCategory.BIOLOGICAL_PROCESS
        assert metadata.execution_phase == ExecutionPhase.MAIN_PROCESSES
        assert "k" in metadata.parameters
        assert metadata.conservation_laws == ["mass"]
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        # Test with empty name - should raise
        with pytest.raises(ValueError):
            FunctorMetadata(
                name="",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                execution_phase=ExecutionPhase.MAIN_PROCESSES
            )
        
        # Test with invalid spatial dimensions
        with pytest.raises(ValueError):
            FunctorMetadata(
                name="Invalid",
                spatial_dimensions=4,  # Should be 0-3
                description="Unit test metadata",
                references=["Unit test reference"],
                output_variables=["invalid_output"],
            )


class TestFunctorBase:
    """Test suite for FunctorBase."""
    
    def test_simple_functor(self):
        """Test a simple decay functor."""
        
        class DecayFunctor(FunctorBase):
            def __init__(self, decay_rate: float = 0.1):
                metadata = FunctorMetadata(
                    name="Decay",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Unit test metadata",
                    mathematical_expression="dC/dt = -k*C",
                    references=["Unit test reference"],
                    parameters={"k": "Decay rate constant [1/s]"},
                    input_variables=["concentration"],
                    output_variables=["decay_flux"]
                )
                config = {"parameters": {"k": decay_rate}}
                super().__init__(name="Decay", config=config, metadata=metadata)
                self.decay_rate = decay_rate
            
            def get_dependencies(self) -> List[str]:
                return []
            
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                concentration = state.get('concentration', torch.tensor(0.0))
                decay_flux = -self.decay_rate * concentration
                return {'decay_flux': decay_flux}
        
        # Create and test functor
        functor = DecayFunctor(decay_rate=0.2)
        
        # Test metadata
        assert functor.name == "Decay"
        assert functor.metadata.name == "Decay"
        
        # Test dependency management
        assert functor.get_dependencies() == []
        
        # Test execution
        state = {'concentration': torch.tensor(100.0)}
        output = functor.forward(state, time=0.0, dt=1.0)
        
        assert 'decay_flux' in output
        expected_flux = -0.2 * 100.0
        assert torch.allclose(output['decay_flux'], torch.tensor(expected_flux))

    def test_validation_context_carries_required_inputs(self):
        """Expose the Functor's required inputs to metadata-derived rules."""

        class OptionalInputFunctor(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="OptionalInputValidationTest",
                    category=ProcessCategory.PHYSICAL_FIELD,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Optional input validation test",
                    mathematical_expression="y = x",
                    references=["Unit test reference"],
                    input_variables=["required_input", "optional_input"],
                    output_variables=["result"],
                )
                super().__init__(
                    name="OptionalInputValidationTest",
                    config={},
                    metadata=metadata,
                )
                self.optional_input_variables = ["optional_input"]

            def get_dependencies(self) -> List[str]:
                return []

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {"result": state["required_input"]}

        class EmptyReport:
            @staticmethod
            def get_results_by_level(level):
                return []

        class CapturingEngine:
            def __init__(self):
                self.context = None

            def run(self, *, context, **kwargs):
                self.context = context
                return EmptyReport()

        functor = OptionalInputFunctor()
        engine = CapturingEngine()
        functor._validation_engine = engine
        result = functor(
            {"required_input": torch.tensor(1.0)},
            time=0.0,
            dt=1.0,
        )

        assert result["result"].item() == 1.0
        assert engine.context is not None
        assert engine.context.metadata["required_input_variables"] == [
            "required_input"
        ]
        assert "optional_input_variables" not in engine.context.metadata
        with pytest.raises(ExecutionError, match="required_input"):
            functor(
                {"optional_input": torch.tensor(1.0)},
                time=0.0,
                dt=1.0,
            )

    def test_unit_rule_checks_only_the_required_inputs(self):
        """Check the required inputs named by the Functor and every required output."""
        metadata = FunctorMetadata(
            name="OptionalUnitRuleTest",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Optional unit rule test",
            mathematical_expression="y = x",
            references=["Unit test reference"],
            input_variables=["required_input", "optional_input"],
            output_variables=[
                "result",
                "required_observation",
                "optional_observation",
            ],
            optional_output_variables=["optional_observation"],
        )
        registry = ValidationRegistry()
        integrator = FunctorMetadataIntegrator(registry=registry)
        integrator.register_functor_metadata(
            "OptionalUnitRuleTest",
            metadata,
            overwrite=True,
        )
        rule = registry.get_rule("OptionalUnitRuleTest_unit_validation")
        assert rule is not None

        valid_results = rule(
            ValidationContext(
                metadata={
                    "required_input_variables": ["required_input"],
                    "optional_output_variables": ["optional_observation"],
                },
                state={"required_input": torch.tensor(1.0)},
                output={
                    "result": torch.tensor(1.0),
                    "required_observation": torch.tensor(1.0),
                },
            )
        )
        assert not any("optional_input" in result.message for result in valid_results)
        assert not any("optional_observation" in result.message for result in valid_results)

        invalid_results = rule(
            ValidationContext(
                metadata={
                    "required_input_variables": ["required_input"],
                    "optional_output_variables": ["optional_observation"],
                },
                state={"optional_input": torch.tensor(1.0)},
                output={"result": torch.tensor(1.0)},
            )
        )
        messages = [result.message for result in invalid_results]
        assert any("Input variable 'required_input' not found" in msg for msg in messages)
        assert any(
            "Output variable 'required_observation' not found" in msg
            for msg in messages
        )
    
    def test_functor_with_dependencies(self):
        """Test functor with dependencies."""
        
        class PhotosynthesisFunctor(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="Photosynthesis",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Unit test metadata",
                    mathematical_expression="GPP = Vcmax * f(light) * f(T) * f(CO2)",
                    references=["Unit test reference"],
                    input_variables=["temperature", "light", "co2"],
                    output_variables=["gpp"]
                )
                config = {"dependencies": ["temperature", "light", "co2"]}
                super().__init__(name="Photosynthesis", config=config, metadata=metadata)
            
            def get_dependencies(self) -> List[str]:
                return ["temperature", "light", "co2"]
            
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                temp = state.get('temperature', torch.tensor(298.15))
                light = state.get('light', torch.tensor(1000.0))
                co2 = state.get('co2', torch.tensor(400.0))
                
                # Simplified photosynthesis
                gpp = 50.0 * (light / 1000.0) * (co2 / 400.0)
                return {'gpp': gpp}
        
        functor = PhotosynthesisFunctor()
        
        # Test dependencies
        deps = functor.get_dependencies()
        assert len(deps) == 3
        assert "temperature" in deps
        assert "light" in deps
        assert "co2" in deps
        
        # Test dependency management
        functor.add_dependency("water_stress")
        assert "water_stress" in functor._dependencies
        
        functor.remove_dependency("water_stress")
        assert "water_stress" not in functor._dependencies
    
    def test_functor_parameter_management(self):
        """Test parameter get/set functionality."""
        
        class ParameterizedFunctor(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="Parameterized",
                    description="Unit test metadata",
                    references=["Unit test reference"],
                    output_variables=["parameterized_output"],
                )
                config = {
                    "parameters": {
                        "param1": 1.0,
                        "param2": {"value": 2.0, "bounds": (0, 10)}
                    }
                }
                super().__init__(name="Parameterized", config=config, metadata=metadata)
            
            def get_dependencies(self) -> List[str]:
                return []
            
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}
        
        functor = ParameterizedFunctor()
        
        # Test parameter retrieval
        assert functor.get_parameter_value("param1") == 1.0
        assert functor.get_parameter_value("param2") == 2.0
        assert functor.get_parameter_value("nonexistent") is None
        
        # Test parameter setting
        functor.set_parameter_value("param1", 3.0)
        assert functor.get_parameter_value("param1") == 3.0
        
        # Test bounded parameter
        with pytest.raises(ValueError):
            functor.set_parameter_value("param2", 11.0)  # Outside bounds

    @staticmethod
    def _instance_parameter_functor(parameter_manager=None):
        class InstanceParameterFunctor(FunctorBase):
            def __init__(self, manager):
                metadata = FunctorMetadata(
                    name="InstanceParameter",
                    description="Unit test metadata",
                    references=["Unit test reference"],
                    output_variables=["instance_parameter_output"],
                )
                kwargs = {} if manager is None else {"parameter_manager": manager}
                super().__init__(
                    name="InstanceParameter",
                    config={"parameters": {"param1": 1.0}},
                    metadata=metadata,
                    **kwargs,
                )

            def get_dependencies(self) -> List[str]:
                return self._merge_declared_dependencies()

            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                return {}

        return InstanceParameterFunctor(parameter_manager)

    def test_dotted_name_is_absolute_only_under_a_declared_namespace(self):
        """A dotted name is read as ``namespace.key`` only when a parameter file
        declares the namespace (or a related one); a database root directory
        alone is not enough, so an undeclared ``transport.<ns>.key`` stays scoped
        to the Functor namespace."""
        functor = self._instance_parameter_functor()
        default_ns = functor._default_parameter_namespace()

        full_key, name, namespace = functor._resolve_parameter_lookup(
            "transport.probe_undeclared_namespace_d53.value", None
        )
        assert namespace == default_ns
        assert full_key == f"{default_ns}.transport.probe_undeclared_namespace_d53.value"

        full_key, name, namespace = functor._resolve_parameter_lookup(
            "transport.numerical.track_conservation", None
        )
        assert (namespace, name) == ("transport.numerical", "track_conservation")

    def test_set_parameter_value_stays_on_the_instance(self):
        """An instance value never writes the global ParameterManager."""
        from relic.parameters import get_global_manager

        functor = self._instance_parameter_functor()
        manager = get_global_manager()
        key = f"{functor._default_parameter_namespace()}.param1"
        spec_before = manager.get_parameter(key)

        functor.set_parameter_value("param1", 3.0)

        assert functor.get_parameter_value("param1") == 3.0
        assert manager.get_parameter(key) is spec_before
        other = self._instance_parameter_functor()
        assert other.get_parameter_value("param1") == 1.0

    def test_set_parameter_value_validates_against_the_yaml_spec(self):
        """The instance value is checked against the YAML spec bounds when one exists."""
        from relic.parameters import ParameterCategory, ParameterSpec

        spec = ParameterSpec(
            key="stub.bounded_rate",
            value=0.5,
            units="s-1",
            dtype="float32",
            bounds=(0.0, 1.0),
            category=ParameterCategory.BIOPHYSICAL,
            description="Bounded rate used to verify instance-value validation.",
            references=["RELIC FunctorBase parameter contract"],
            tags={"unit-test"},
            provenance={"source": "test", "version": "1.0.0"},
            requires_grad=False,
        )

        class StubManager:
            def get_parameter(self, key):
                return spec if key.endswith(".bounded_rate") else None

        functor = self._instance_parameter_functor(StubManager())
        with pytest.raises(ValueError, match="bounded_rate"):
            functor.set_parameter_value("bounded_rate", 2.0)
        functor.set_parameter_value("bounded_rate", 0.25)
        assert functor.parameter_values["bounded_rate"] == 0.25

    def test_functor_conservation(self):
        """Test conservation law checking."""
        
        class ConservativeFunctor(FunctorBase):
            def __init__(self, check_conservation: bool = True):
                metadata = FunctorMetadata(
                    name="Conservative",
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Unit test metadata",
                    references=["Unit test reference"],
                    conservation_laws=["mass", "energy"],
                    output_variables=["carbon_pool1_flux", "carbon_pool2_flux"],
                )
                config = {"check_conservation": check_conservation}
                super().__init__(name="Conservative", config=config, metadata=metadata)
                self.check_conservation = check_conservation
            
            def get_dependencies(self) -> List[str]:
                return []
            
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                # Conservative process - mass flows from pool1 to pool2
                pool1 = state.get('carbon_pool1', torch.tensor(100.0))
                transfer = 10.0
                
                if self.check_conservation:
                    # Conservative - sum of changes is zero
                    return {
                        'carbon_pool1_flux': -transfer,
                        'carbon_pool2_flux': transfer
                    }
                else:
                    # Non-conservative - mass is lost
                    return {
                        'carbon_pool1_flux': -transfer,
                        'carbon_pool2_flux': transfer * 0.9  # 10% loss
                    }
        
        # Test conservative functor
        functor = ConservativeFunctor(check_conservation=True)
        state = {'carbon_pool1': torch.tensor(100.0), 'carbon_pool2': torch.tensor(50.0)}
        
        output = functor.forward(state, time=0.0, dt=1.0)
        violations = functor.check_conservation_laws(state, output)
        
        # Should have near-zero violation for mass
        assert 'mass' in violations
        assert violations['mass'] < 1e-8
        
        # Test non-conservative functor
        functor_bad = ConservativeFunctor(check_conservation=False)
        output_bad = functor_bad.forward(state, time=0.0, dt=1.0)
        violations_bad = functor_bad.check_conservation_laws(state, output_bad)
        
        # Should detect violation
        assert violations_bad['mass'] > 0.9  # 1.0 mass violation
    
    def test_functor_error_handling(self):
        """Test error handling and recovery."""
        
        class ErrorProneFunctor(FunctorBase):
            def __init__(self, should_fail: bool = False):
                metadata = FunctorMetadata(
                    name="ErrorProne",
                    description="Unit test metadata",
                    references=["Unit test reference"],
                    output_variables=["output"],
                )
                config = {"should_fail": should_fail}
                super().__init__(name="ErrorProne", config=config, metadata=metadata)
                self.should_fail = should_fail
            
            def get_dependencies(self) -> List[str]:
                return []
            
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                if self.should_fail:
                    raise RuntimeError("Simulated failure")
                return {'output': torch.tensor(1.0)}
        
        # Test normal execution
        functor = ErrorProneFunctor(should_fail=False)
        state = {}
        
        output = functor(state, time=0.0, dt=1.0)
        assert 'output' in output
        assert functor.state == FunctorState.COMPLETED
        assert functor._error_count == 0
        
        # Test error handling
        functor_fail = ErrorProneFunctor(should_fail=True)
        
        with pytest.raises(ExecutionError):
            functor_fail(state, time=0.0, dt=1.0)
        
        assert functor_fail.state == FunctorState.FAILED
        assert functor_fail._error_count == 1
        assert functor_fail._last_error is not None
    
    def test_functor_execution_context(self):
        """Test execution context and statistics."""
        
        class ContextAwareFunctor(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="ContextAware",
                    description="Unit test metadata",
                    references=["Unit test reference"],
                    output_variables=["output"],
                )
                config = {}
                super().__init__(name="ContextAware", config=config, metadata=metadata)
            
            def get_dependencies(self) -> List[str]:
                return []
            
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                # Scientific inputs come from the state only.
                output = torch.tensor(1.0)
                if 'scale_factor' in state:
                    output = output * state['scale_factor']

                return {'output': output}
        
        functor = ContextAwareFunctor()
        state = {}
        
        # Test without context
        output1 = functor(state, time=0.0, dt=1.0)
        assert torch.allclose(output1['output'], torch.tensor(1.0))
        
        # The scale factor is a state input
        output2 = functor({'scale_factor': torch.tensor(2.0)}, time=1.0, dt=1.0)
        assert torch.allclose(output2['output'], torch.tensor(2.0))
        
        # Check execution statistics
        stats = functor.get_execution_stats()
        assert stats['execution_count'] == 2
        assert stats['state'] == FunctorState.COMPLETED.value
        assert stats['total_execution_time'] > 0
        
        # Reset and verify
        functor.reset_stats()
        stats_reset = functor.get_execution_stats()
        assert stats_reset['execution_count'] == 0


class _ProbeFunctor(FunctorBase):
    def __init__(self, name="error_probe", conservation_laws=None, config=None):
        metadata = FunctorMetadata(
            name=name,
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Error propagation probe",
            mathematical_expression="y = x",
            references=["Unit test reference"],
            input_variables=["x"],
            output_variables=["y"],
            conservation_laws=conservation_laws or [],
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> List[str]:
        return []

    def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
        return {"y": state["x"]}


class _RejectingManager:
    parameters: Dict = {}

    def get_parameter(self, key):
        return None

    def get(self, key, **kwargs):
        return None

    def register_parameter(self, spec):
        raise RuntimeError("parameter registry is read-only")


class TestFunctorBaseErrorPropagation:
    """FunctorBase paths raise instead of logging and continuing."""

    def test_validation_rule_registration_failure_propagates(self, monkeypatch):
        def _broken(*_args, **_kwargs):
            raise RuntimeError("rule registry unavailable")

        monkeypatch.setattr(functor_integration, "register_functor_validations", _broken)
        with pytest.raises(RuntimeError, match="rule registry unavailable"):
            _ProbeFunctor()

    def test_parameter_manager_failure_is_not_read_as_no_manager(self, monkeypatch):
        functor = _ProbeFunctor(config={"parameters": {"rate": 2.0}})

        def _broken():
            raise RuntimeError("parameter database unavailable")

        monkeypatch.setattr(functor, "get_parameter_manager", _broken)
        with pytest.raises(RuntimeError, match="parameter database unavailable"):
            functor.get_parameter_value("rate", source_order=("config",))
        with pytest.raises(RuntimeError, match="parameter database unavailable"):
            functor.get_parameter_value("ns.rate", source_order=("config",))

    def test_requested_parameter_registration_failures_propagate(self):
        functor = _ProbeFunctor(
            config={"parameters": {"rate": {
                "value": 2.0, "units": "s-1", "dtype": "float64", "bounds": [0.0, 10.0],
            }}}
        )
        functor.parameter_manager = _RejectingManager()

        with pytest.raises(RuntimeError, match="read-only"):
            functor.register_parameters_to_manager()

    def test_registering_a_numeric_config_parameter_without_bounds_is_rejected(self):
        """A numeric parameter declares an interval (parameter_bounds_error)."""
        functor = _ProbeFunctor(
            config={"parameters": {"rate": {"value": 2.0, "units": "s-1", "dtype": "float64"}}}
        )
        functor.parameter_manager = _RejectingManager()
        with pytest.raises(ValueError, match="rate: bounds is None but dtype 'float64' is numeric"):
            functor.register_parameters_to_manager()

    def test_registering_a_config_parameter_without_dtype_is_rejected(self):
        """dtype has no universal default: a registered parameter declares it."""
        functor = _ProbeFunctor(config={"parameters": {"rate": {"value": 2.0, "units": "s-1"}}})
        functor.parameter_manager = _RejectingManager()
        with pytest.raises(ValueError, match="rate.*dtype"):
            functor.register_parameters_to_manager()

    def test_config_parameter_entry_must_declare_a_value(self):
        from relic.core.functor import ConfigurationError

        with pytest.raises(ConfigurationError, match="value"):
            _ProbeFunctor(config={"parameters": {"rate": {"units": "s-1", "bounds": [0.0, 10.0]}}})

    def test_undeclared_config_units_are_not_invented(self):
        """No unit is recorded without a declaration; registering then raises."""
        functor = _ProbeFunctor(config={"parameters": {"rate": {"value": 2.0}}})
        assert "rate" not in functor._parameter_units
        assert "rate" not in functor._parameter_bounds
        functor.parameter_manager = _RejectingManager()
        with pytest.raises(ValueError, match="rate.*units"):
            functor.register_parameters_to_manager()

    def test_config_parameter_bounds_come_from_the_entry_or_the_database_spec(self, monkeypatch):
        from relic.parameters import ParameterSpec

        declared = _ProbeFunctor(config={"parameters": {"rate": {"value": 2.0, "units": "s-1", "bounds": [0.0, 5.0]}}})
        assert declared._parameter_bounds["rate"] == (0.0, 5.0)

        class _SpecManager:
            parameters: Dict = {}

            def get_parameter(self, key):
                if key.endswith(".rate"):
                    return ParameterSpec(key=key, value=1.0, units="s-1", dtype="float64", bounds=(0.5, 3.0), category="model_parameter")
                return None

            def get(self, key, **kwargs):
                return None

        monkeypatch.setattr(FunctorBase, "get_parameter_manager", lambda self: _SpecManager())
        from_spec = _ProbeFunctor(name="bounds_from_spec", config={"parameters": {"rate": {"value": 2.0, "units": "s-1"}}})
        assert from_spec._parameter_bounds["rate"] == (0.5, 3.0)
        canonical = _ProbeFunctor(name="units_from_spec", config={"parameters": {"rate": {"value": 2.0}}})
        assert canonical._parameter_units["rate"] == "s-1"
        assert canonical._parameter_bounds["rate"] == (0.5, 3.0)

        class _EmptyManager(_SpecManager):
            def get_parameter(self, key):
                return None

        monkeypatch.setattr(FunctorBase, "get_parameter_manager", lambda self: _EmptyManager())
        undeclared = _ProbeFunctor(name="no_bounds", config={"parameters": {"rate": {"value": 2.0, "units": "s-1"}}})
        assert "rate" not in undeclared._parameter_bounds

    def test_registering_a_config_parameter_without_units_is_rejected(self):
        functor = _ProbeFunctor(config={"parameters": {"rate": 2.0}})
        functor.parameter_manager = _RejectingManager()
        with pytest.raises(ValueError, match="rate.*units"):
            functor.register_parameters_to_manager()

    @pytest.mark.parametrize("strict_value", [torch.tensor(1.0), 1, "true"])
    def test_forward_conservation_strict_flag_must_be_a_bool(self, monkeypatch, strict_value):
        functor = _ProbeFunctor(name="strict_probe", conservation_laws=["mass"])

        def _parameter(name, **kwargs):
            return 1.0e-6 if name == "conservation_tolerance" else strict_value

        monkeypatch.setattr(functor, "get_parameter_value", _parameter)
        with pytest.raises(TypeError, match="strict_conservation"):
            functor._enforce_forward_conservation({"x": torch.tensor(1.0)}, {"y": torch.tensor(5.0)})

    def test_local_conservation_rejects_a_non_numeric_declared_output(self):
        functor = _ProbeFunctor(conservation_laws=["mass"])
        with pytest.raises(TypeError, match="numeric"):
            functor.check_conservation_laws({}, {"y": "not a number"})
        assert functor.check_conservation_laws({}, {"y": torch.tensor([1.0, -0.25])}) == {"mass": 0.75}

    def test_validation_engine_failures_fail_the_call(self, monkeypatch):
        import relic.validation.framework.engine as engine_module

        class _BrokenEngine:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("validation engine cannot start")

        monkeypatch.setattr(engine_module, "ValidationEngine", _BrokenEngine)
        functor = _ProbeFunctor()
        with pytest.raises(ExecutionError) as error:
            functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0)
        assert "validation engine cannot start" in str(error.value.__cause__)

        class _FailingRunEngine:
            def run(self, **kwargs):
                raise RuntimeError("validation run crashed")

        functor = _ProbeFunctor(name="error_probe_run")
        functor._validation_engine = _FailingRunEngine()
        with pytest.raises(ExecutionError) as error:
            functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0)
        assert "validation run crashed" in str(error.value.__cause__)


@pytest.fixture
def forced_validation_findings():
    """Register L1 rules, tagged with a probe functor name, that report findings."""
    from relic.validation.framework.registry import RuleMetadata

    registry = get_global_registry()
    registered = []

    def _register(functor_name, *, error=True, warning=True, error_message=None):
        def forced_rule(context):
            results = []
            if error:
                functor_integration._add_error(
                    results,
                    message=error_message or f"forced error for {functor_name}",
                    category="probe",
                    layer="L1",
                    scope=functor_name,
                )
            if warning:
                functor_integration._add_warning(
                    results,
                    message=f"forced warning for {functor_name}",
                    category="probe",
                    layer="L1",
                    scope=functor_name,
                )
            return results

        rule_id = f"{functor_name}_forced_findings"
        registry.register_rule(
            forced_rule,
            RuleMetadata(
                rule_id=rule_id,
                name="forced findings",
                description="Probe rule for FunctorBase validation strictness",
                layer="L1",
                tags=[functor_name],
                enabled=True,
            ),
            overwrite=True,
        )
        registered.append(rule_id)

    yield _register
    for rule_id in registered:
        registry.unregister_rule(rule_id)


class TestCallArgumentForms:
    """time and dt: two positional arguments or keywords; no environment argument."""

    def test_positional_and_keyword_time_step_both_work(self):
        functor = _ProbeFunctor(name="call_form_probe")
        assert torch.equal(functor({"x": torch.tensor(1.0)}, 0.0, 1.0)["y"], torch.tensor(1.0))
        assert torch.equal(functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0)["y"], torch.tensor(1.0))

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            (({"t": 1.0},), {"time": 0.0, "dt": 1.0}),
            (({"t": 1.0}, 0.0, 1.0), {}),
            ((0.0,), {"dt": 1.0}),
            ((0.0, 1.0), {"time": 0.0}),
        ],
    )
    def test_other_positional_forms_raise(self, args, kwargs):
        functor = _ProbeFunctor(name="call_form_reject_probe")
        with pytest.raises(TypeError, match="time"):
            functor({"x": torch.tensor(1.0)}, *args, **kwargs)

    @pytest.mark.parametrize("environment", [{}, {"t": 1.0}, None])
    def test_an_environment_keyword_is_rejected(self, environment):
        """The state is the only input channel: no environment keyword reaches forward."""
        functor = _ProbeFunctor(name="call_form_environment_probe")
        with pytest.raises(TypeError, match="environment.*state"):
            functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0, environment=environment)


class TestParameterValuesHaveNoCallerDefault:
    """YAML (manager) or config supply a value; a caller never does."""

    @pytest.mark.parametrize("owner", ["FunctorBase", "BiologicalProcessFunctor", "ScaleCouplingFunctor"])
    def test_the_read_path_has_no_default_route(self, owner):
        import inspect

        import relic.core.functor as functor_package

        parameters = inspect.signature(getattr(functor_package, owner).get_parameter_value).parameters
        for removed in ("default", "default_value", "register_if_missing"):
            assert removed not in parameters, (owner, removed)
        assert parameters["source_order"].default == ("manager", "config")
        assert not any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values()), owner

    @pytest.mark.parametrize("order", [("manager", "default"), ("default",), ("config", "local"), ()])
    def test_source_order_names_only_manager_and_config(self, order):
        functor = _ProbeFunctor(name="source_order_probe", config={"parameters": {"rate": {"value": 2.0, "units": "s-1"}}})
        with pytest.raises(ValueError, match="source_order"):
            functor.get_parameter_value("rate", source_order=order)

    def test_a_missing_parameter_is_none_or_raises(self):
        functor = _ProbeFunctor(name="missing_probe")
        assert functor.get_parameter_value("no_such_rate", source_order=("config",)) is None
        with pytest.raises(ValueError, match="no_such_rate"):
            functor.get_parameter_value("no_such_rate", source_order=("config",), required=True)
        with pytest.raises(TypeError, match="default"):
            functor.get_parameter_value("no_such_rate", default=1.0)


class TestFunctorBaseValidationStrictness:
    """Framework validation errors fail the call unless non-strict is requested
    explicitly; findings go to the caller's validation_report, never only to the
    log, and the Functor keeps no findings state."""

    def test_validation_errors_fail_the_call_by_default(self, forced_validation_findings):
        functor = _ProbeFunctor(name="strict_default_probe")
        forced_validation_findings("strict_default_probe")
        report = []

        with pytest.raises(ExecutionError, match="forced error for strict_default_probe"):
            functor({"x": torch.tensor(1.0)}, time=2.0, dt=0.5, validation_report=report)

        assert len(report) == 1
        record = report[0]
        assert record["functor"] == "strict_default_probe"
        assert record["time"] == 2.0
        assert record["dt"] == 0.5
        assert record["strict_validation"] is True
        assert record["disposition"] == "raised"
        assert [entry["message"] for entry in record["errors"]] == [
            "forced error for strict_default_probe"
        ]
        assert record["errors"][0]["level"] == "ERROR"
        assert [entry["message"] for entry in record["warnings"]] == [
            "forced warning for strict_default_probe"
        ]

    def test_non_strict_is_an_explicit_opt_in_and_downgrades_are_recorded(
        self, forced_validation_findings
    ):
        functor = _ProbeFunctor(name="non_strict_probe", config={"strict_validation": False})
        forced_validation_findings("non_strict_probe")
        report = []

        updates = functor({"x": torch.tensor(3.0)}, time=1.0, dt=1.0, validation_report=report)
        functor({"x": torch.tensor(4.0)}, time=2.0, dt=1.0, validation_report=report)

        assert torch.equal(updates["y"], torch.tensor(3.0))
        assert [record["time"] for record in report] == [1.0, 2.0]
        assert all(record["strict_validation"] is False for record in report)
        assert all(record["disposition"] == "downgraded" for record in report)
        assert report[0]["errors"][0]["message"] == "forced error for non_strict_probe"

    @pytest.mark.parametrize(
        "message",
        [
            "Required parameter 'probe_rate' not found",
            "Parameter 'foreign_rate' value 5.0 above maximum 1.0",
        ],
    )
    def test_strict_validation_raises_parameter_findings_too(
        self, forced_validation_findings, message
    ):
        """No parameter finding is exempt from strict validation."""
        functor = _ProbeFunctor(name="parameter_finding_probe")
        forced_validation_findings("parameter_finding_probe", warning=False, error_message=message)
        report = []

        with pytest.raises(ExecutionError, match=message.split("'")[1]):
            functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0, validation_report=report)
        assert [record["disposition"] for record in report] == ["raised"]

    def test_a_downgrade_needs_a_report_to_receive_it(self, forced_validation_findings):
        """Without a caller report the downgrade would only reach the log, so the
        errors are raised."""
        functor = _ProbeFunctor(name="unreported_probe", config={"strict_validation": False})
        forced_validation_findings("unreported_probe")

        with pytest.raises(ExecutionError, match="validation_report"):
            functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0)

    def test_warnings_alone_are_recorded_and_do_not_fail(self, forced_validation_findings):
        functor = _ProbeFunctor(name="warning_only_probe")
        forced_validation_findings("warning_only_probe", error=False)
        report = []

        functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0, validation_report=report)
        functor({"x": torch.tensor(1.0)}, time=1.0, dt=1.0)

        (record,) = report
        assert record["disposition"] == "passed_with_warnings"
        assert record["errors"] == []
        assert record["warnings"][0]["message"] == "forced warning for warning_only_probe"

    def test_calls_without_findings_leave_the_report_empty(self):
        functor = _ProbeFunctor(name="clean_probe")
        report = []
        functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0, validation_report=report)
        assert report == []

    @pytest.mark.parametrize("report", [(), {}, "findings"])
    def test_validation_report_must_be_a_list(self, report):
        functor = _ProbeFunctor(name="bad_report_probe")
        with pytest.raises(TypeError, match="validation_report"):
            functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0, validation_report=report)

    def test_the_functor_keeps_no_validation_state(self, forced_validation_findings):
        """Strictness is read from the config and findings leave with the call:
        the instance holds only the runtime shell the graph executor accepts for
        deterministic_pure nodes."""
        from relic.orchestration.executor import _CACHE_RUNTIME_SHELL_FIELDS

        functor = _ProbeFunctor(name="stateless_probe", config={"strict_validation": False})
        forced_validation_findings("stateless_probe")
        before = set(vars(functor))
        functor({"x": torch.tensor(1.0)}, time=0.0, dt=1.0, validation_report=[])

        assert set(vars(functor)) == before
        assert before <= _CACHE_RUNTIME_SHELL_FIELDS, sorted(before - _CACHE_RUNTIME_SHELL_FIELDS)

    @pytest.mark.parametrize("flag", [0, 1, "false", None, torch.tensor(False)])
    def test_strict_validation_must_be_a_bool(self, flag):
        with pytest.raises(TypeError, match="strict_validation"):
            _ProbeFunctor(name="bad_flag_probe", config={"strict_validation": flag})

    @pytest.mark.parametrize("flag", [True, False])
    def test_removed_strict_validation_errors_key_is_rejected(self, flag):
        from relic.core.functor import ConfigurationError

        with pytest.raises(ConfigurationError, match="strict_validation_errors"):
            _ProbeFunctor(name="legacy_flag_probe", config={"strict_validation_errors": flag})


class TestFunctorIntegration:
    """Test functor integration with state management."""

    def test_parameter_rule_uses_resolved_spec_without_namespace_value_scan(
        self,
        monkeypatch,
    ):
        """A resolved YAML spec must supply its value without a generic tree lookup."""
        class Spec:
            value = 2.5
            bounds = (0.0, 5.0)
            units = "s-1"

        class Manager:
            def __init__(self):
                self.get_calls = 0

            def get_parameter(self, key):
                return Spec() if key == "rate" else None

            def get(self, *args, **kwargs):
                self.get_calls += 1
                raise AssertionError("generic manager.get must not be used")

        manager = Manager()
        monkeypatch.setattr(
            functor_integration,
            "_resolve_validation_manager",
            lambda: manager,
        )
        metadata = FunctorMetadata(
            name="SpecValueValidation",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Parameter validation lookup test",
            references=["Unit test reference"],
            parameters={
                "rate": {
                    "required": True,
                    "min": 0.0,
                    "max": 5.0,
                    "units": "s-1",
                }
            },
            output_variables=["result"],
        )
        registry = ValidationRegistry()
        integrator = FunctorMetadataIntegrator(registry=registry)
        integrator.register_functor_metadata(
            "SpecValueValidation",
            metadata,
            overwrite=True,
        )
        rule = registry.get_rule("SpecValueValidation_parameter_validation")
        assert rule is not None

        results = rule(ValidationContext(parameters={}))

        assert results == []
        assert manager.get_calls == 0

    def test_parameter_rule_missing_required_spec_fails_without_tree_scan(
        self,
        monkeypatch,
    ):
        """Missing required YAML specs must fail closed without namespace expansion."""
        class Manager:
            def __init__(self):
                self.get_calls = 0

            def get_parameter(self, key):
                return None

            def get(self, *args, **kwargs):
                self.get_calls += 1
                raise AssertionError("generic manager.get must not be used")

        manager = Manager()
        monkeypatch.setattr(
            functor_integration,
            "_resolve_validation_manager",
            lambda: manager,
        )
        metadata = FunctorMetadata(
            name="MissingSpecValidation",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Missing parameter validation lookup test",
            references=["Unit test reference"],
            parameters={"missing_rate": {"required": True}},
            output_variables=["result"],
        )
        registry = ValidationRegistry()
        integrator = FunctorMetadataIntegrator(registry=registry)
        integrator.register_functor_metadata(
            "MissingSpecValidation",
            metadata,
            overwrite=True,
        )
        rule = registry.get_rule("MissingSpecValidation_parameter_validation")
        assert rule is not None

        results = rule(ValidationContext(parameters={}))

        assert any(
            "Required parameter 'missing_rate' not found" in result.message
            for result in results
        )
        assert manager.get_calls == 0

    @staticmethod
    def _parameter_rule(name):
        metadata = FunctorMetadata(
            name=name,
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Parameter validation error propagation test",
            references=["Unit test reference"],
            parameters={"rate": {"required": False}},
            output_variables=["result"],
        )
        registry = ValidationRegistry()
        integrator = FunctorMetadataIntegrator(registry=registry)
        integrator.register_functor_metadata(name, metadata, overwrite=True)
        rule = registry.get_rule(f"{name}_parameter_validation")
        assert rule is not None
        return rule

    def test_parameter_rule_propagates_a_failing_parameter_manager(self, monkeypatch):
        """A manager that fails to load is an error, not an absent manager."""

        def failing_manager():
            raise RuntimeError("parameter database failed to load")

        monkeypatch.setattr(functor_integration, "get_global_manager", failing_manager)
        rule = self._parameter_rule("FailingManagerValidation")
        with pytest.raises(RuntimeError, match="parameter database failed to load"):
            rule(ValidationContext(parameters={}))

    def test_parameter_rule_propagates_a_failing_spec_lookup(self, monkeypatch):
        """A spec lookup error (for example a unit failure) is not a missing spec."""

        class Manager:
            def get_parameter(self, key):
                raise ValueError(f"units of {key} failed validation")

        manager = Manager()
        monkeypatch.setattr(functor_integration, "_resolve_validation_manager", lambda: manager)
        rule = self._parameter_rule("FailingSpecValidation")
        with pytest.raises(ValueError, match="units of rate failed validation"):
            rule(ValidationContext(parameters={}))

    def test_biological_rule_does_not_claim_a_default_for_a_missing_input(self):
        """No Functor substitutes a default for a missing biological input."""
        metadata = FunctorMetadata(
            name="BiologicalInputValidation",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Biological input message test",
            references=["Unit test reference"],
            input_variables=["par"],
            output_variables=["result"],
        )
        registry = ValidationRegistry()
        integrator = FunctorMetadataIntegrator(registry=registry)
        integrator.register_functor_metadata(
            "BiologicalInputValidation", metadata, overwrite=True
        )
        rule = registry.get_rule("BiologicalInputValidation_biological_constraints")
        assert rule is not None
        messages = [result.message for result in rule(ValidationContext(state={}))]
        assert any("'par'" in message for message in messages)
        assert not any("using default" in message for message in messages)
        assert any("no default is substituted" in message for message in messages)

    def test_biological_rule_reads_the_ssot_temperature_range(self):
        """The biological range is validation.numerical.clamp_min/max_temperature."""
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        key = "validation.numerical.clamp_max_temperature"
        original = float(manager.get(key, required=True))
        metadata = FunctorMetadata(
            name="BiologicalRangeValidation",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Biological temperature range test",
            references=["Unit test reference"],
            input_variables=["temperature"],
            output_variables=["result"],
        )
        registry = ValidationRegistry()
        FunctorMetadataIntegrator(registry=registry).register_functor_metadata(
            "BiologicalRangeValidation", metadata, overwrite=True
        )
        rule = registry.get_rule("BiologicalRangeValidation_biological_constraints")
        state = {"temperature": torch.tensor(310.0)}
        try:
            manager.set_value(key, 300.0, validate=True)
            messages = [result.message for result in rule(ValidationContext(state=state))]
            assert any("outside biological range" in message for message in messages)
        finally:
            manager.set_value(key, original, validate=True)

    def test_conservation_laws_register_no_name_based_flux_rule(self):
        """Conservation belongs to the documented layers (FunctorBase local hook,
        StateManager ledgers, ConservationGuardian); metadata registers no
        name-substring flux rule with its own tolerance."""
        metadata = FunctorMetadata(
            name="ConservationLawRule",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Conservation rule registration test",
            references=["Unit test reference"],
            input_variables=["carbon_pool"],
            output_variables=["carbon_pool"],
            conservation_laws=["carbon"],
        )
        registry = ValidationRegistry()
        rule_ids = FunctorMetadataIntegrator(registry=registry).register_functor_metadata(
            "ConservationLawRule", metadata, overwrite=True
        )
        assert registry.get_rule("ConservationLawRule_conservation_validation") is None
        assert not [rule_id for rule_id in rule_ids if rule_id.endswith("_conservation_validation")]

    def test_global_integrator_rebinds_after_registry_reset(self):
        """Keep generated rules attached to the current canonical registry.

        test/validation/conftest.py puts the registry and the integrator that
        were global before the test back afterwards.
        """
        reset_global_registry()
        original_registry = get_global_registry()
        original_integrator = get_global_integrator()
        assert original_integrator.registry is original_registry

        reset_global_registry()
        current_registry = get_global_registry()
        current_integrator = get_global_integrator()

        metadata = FunctorMetadata(
            name="RegistryResetFunctor",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Registry reset integration test",
            mathematical_expression="y = x",
            references=["Unit test reference"],
            input_variables=["x"],
            output_variables=["y"],
        )
        rule_ids = register_functor_validations(
            "RegistryResetFunctor",
            metadata,
            overwrite=True,
        )

        assert current_integrator is not original_integrator
        assert current_integrator.registry is current_registry
        assert rule_ids
        assert all(current_registry.get_rule(rule_id) is not None for rule_id in rule_ids)
    
    def test_functor_state_update(self):
        """Test that functor outputs update state correctly."""
        
        class StateUpdaterFunctor(FunctorBase):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="StateUpdater",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    execution_phase=ExecutionPhase.MAIN_PROCESSES,
                    description="Unit test metadata",
                    references=["Unit test reference"],
                    input_variables=["temperature", "moisture"],
                    output_variables=["evaporation", "cooling"]
                )
                config = {}
                super().__init__(name="StateUpdater", config=config, metadata=metadata)
            
            def get_dependencies(self) -> List[str]:
                return []
            
            def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
                temp = state.get('temperature', torch.tensor(300.0))
                moisture = state.get('moisture', torch.tensor(0.5))
                
                # Simple evaporation model
                evap_rate = 0.1 * moisture * (temp - 273.15) / 30.0
                cooling_rate = -evap_rate * 2.0  # Evaporative cooling
                
                return {
                    'evaporation': evap_rate,
                    'cooling': cooling_rate
                }
        
        functor = StateUpdaterFunctor()
        
        # Initial state
        state = {
            'temperature': torch.tensor(300.0),  # 27 C
            'moisture': torch.tensor(0.5)
        }
        
        # Execute functor
        updates = functor(state, time=0.0, dt=1.0)
        
        # Verify outputs
        assert 'evaporation' in updates
        assert 'cooling' in updates
        
        # Evaporation should be positive
        assert updates['evaporation'] > 0
        
        # Cooling should be negative (temperature decrease)
        assert updates['cooling'] < 0
        
        # Conservation: energy lost to evaporation equals cooling
        assert torch.allclose(updates['cooling'], -2.0 * updates['evaporation'])


def test_functor_composition():
    """Test composition of multiple functors."""
    
    class ProducerFunctor(FunctorBase):
        """Produces a resource."""
        def __init__(self, production_rate: float = 1.0):
            metadata = FunctorMetadata(
                name="Producer",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                description="Unit test metadata",
                references=["Unit test reference"],
                output_variables=["production"]
            )
            config = {"production_rate": production_rate}
            super().__init__(name="Producer", config=config, metadata=metadata)
            self.production_rate = production_rate
        
        def get_dependencies(self) -> List[str]:
            return []
        
        def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
            return {'resource_production': torch.tensor(self.production_rate)}
    
    class ConsumerFunctor(FunctorBase):
        """Consumes a resource."""
        def __init__(self, consumption_rate: float = 0.5):
            metadata = FunctorMetadata(
                name="Consumer",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                description="Unit test metadata",
                references=["Unit test reference"],
                input_variables=["resource"],
                output_variables=["consumption"]
            )
            config = {"consumption_rate": consumption_rate}
            super().__init__(name="Consumer", config=config, metadata=metadata)
            self.consumption_rate = consumption_rate
        
        def get_dependencies(self) -> List[str]:
            return ["Producer"]  # Depends on producer
        
        def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
            resource = state.get('resource', torch.tensor(0.0))
            consumption = torch.min(resource, torch.tensor(self.consumption_rate))
            return {'resource_consumption': -consumption}
    
    # Create functors
    producer = ProducerFunctor(production_rate=2.0)
    consumer = ConsumerFunctor(consumption_rate=1.5)
    
    # Check dependencies
    assert consumer.get_dependencies() == ["Producer"]
    
    # Simulate execution sequence
    state = {'resource': torch.tensor(10.0)}
    
    # Producer adds resource
    producer_output = producer(state, time=0.0, dt=1.0)
    assert producer_output['resource_production'] == 2.0
    
    # Update state (normally done by simulator)
    state['resource'] += producer_output['resource_production']
    
    # Consumer removes resource
    consumer_output = consumer(state, time=0.0, dt=1.0)
    assert consumer_output['resource_consumption'] == -1.5
    
    # Net change
    net_change = producer_output['resource_production'] + consumer_output['resource_consumption']
    assert torch.allclose(net_change, torch.tensor(0.5))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

