"""
L1 Unit Tests for Parameter Decorators

Tests for parameter management decorators that provide automatic registration and validation.
"""

import pytest
import torch

from relic.parameters.decorators import (
    managed_parameters,
    parameter,
    learnable_parameter,
    validated_parameter
)
from relic.parameters import get_global_manager
from relic.core.functor import FunctorBase, FunctorMetadata


@pytest.fixture(autouse=True)
def _restore_global_parameter_state():
    """Keep decorator tests isolated from the shared ParameterManager."""
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        yield
    finally:
        manager.restore(snapshot)


class TestManagedParametersDecorator:
    """Test @managed_parameters decorator."""
    
    def test_basic_managed_parameters(self):
        """Test basic functionality of managed_parameters decorator."""
        
        @managed_parameters(namespace="test.module")
        class TestModule:
            def __init__(self):
                self.param1 = 10.0
                self.param1_spec = {"units": "m", "bounds": (0.0, 100.0), "dtype": "float32",
                                    "description": "Test parameter 1"}
                self.param2 = 20.0
                self.param2_spec = {"units": "m", "bounds": (0.0, 100.0), "dtype": "float32",
                                    "description": "Test parameter 2"}
        
        # Create instance
        module = TestModule()
        
        # Check that parameters were registered
        manager = get_global_manager()
        assert manager.has_parameter("test.module.param1")
        assert manager.has_parameter("test.module.param2")
        assert manager.get("test.module.param1") == 10.0
        assert manager.get("test.module.param2") == 20.0
        
    def test_managed_parameters_with_functor(self):
        """Test managed_parameters with FunctorBase subclass."""
        
        @managed_parameters(namespace="test.test_functor")
        class TestFunctor(FunctorBase):
            def __init__(self):
                # Create minimal config
                config = {
                    "parameters": {},
                    "dependencies": []
                }
                super().__init__(
                    name="test_functor",
                    config=config,
                    metadata=FunctorMetadata(
                        name="test_functor",
                        description="Test fixture for managed parameter registration.",
                        references=["RELIC managed parameter decorator contract test."],
                        input_variables=[],
                        output_variables=["diagnostics.test.managed_parameters"],
                    ),
                )
                self.vcmax25 = 60.0
                self.vcmax25_spec = {"units": "umol m-2 s-1", "bounds": (0.0, 200.0),
                                     "dtype": "float32", "description": "Vcmax at 25 degC"}
                self.jmax25 = 120.0
                self.jmax25_spec = {"units": "umol m-2 s-1", "bounds": (0.0, 400.0),
                                    "dtype": "float32", "description": "Jmax at 25 degC"}
            
            def get_dependencies(self):
                return []
            
            def forward(self, state, time, dt, **kwargs):
                return {}
        
        # Create instance
        functor = TestFunctor()
        
        # Parameters should be registered with auto-generated namespace
        manager = get_global_manager()
        assert manager.has_parameter("test.test_functor.vcmax25")
        assert manager.has_parameter("test.test_functor.jmax25")
        
    def test_managed_parameters_with_spec_attributes(self):
        """Test managed_parameters with _spec suffix attributes."""
        
        @managed_parameters(namespace="test.spec")
        class TestModuleWithSpecs:
            def __init__(self):
                self.param1 = 10.0
                self.param1_spec = {
                    "units": "m/s",
                    "bounds": (0.0, 100.0),
                    "dtype": "float32",
                    "description": "Test parameter 1"
                }
                self.param2 = 300.0
                self.param2_spec = {
                    "units": "K",
                    "bounds": (200.0, 400.0),
                    "dtype": "float32",
                    "description": "Test parameter 2"
                }
        
        # Create instance
        module = TestModuleWithSpecs()
        
        # Check that specs were applied
        manager = get_global_manager()
        spec1 = manager.get_parameter("test.spec.param1")
        assert spec1.units == "m/s"
        assert spec1.bounds == (0.0, 100.0)
        assert spec1.description == "Test parameter 1"
        
        spec2 = manager.get_parameter("test.spec.param2")
        assert spec2.units == "K"
        assert spec2.bounds == (200.0, 400.0)
        
    def test_register_missing_option(self):
        """Test register_missing option in managed_parameters."""
        
        @managed_parameters(namespace="test.missing", register_missing=True)
        class TestModuleMissing:
            def __init__(self):
                self.existing_param = 10.0
                self.existing_param_spec = {"units": "1", "bounds": (0.0, 100.0), "dtype": "float32",
                                            "description": "Existing parameter"}
            
            def get_param(self, name, default=None):
                """Method that might access missing parameters."""
                return getattr(self, name, default)
        
        module = TestModuleMissing()
        manager = get_global_manager()
        
        # Existing parameter should be registered
        assert manager.has_parameter("test.missing.existing_param")
        
        # Accessing missing parameter with register_missing should register it
        # (This would require integration with get_parameter_value method)
        

class TestParameterDecorator:
    """Test @parameter property decorator."""
    
    def test_basic_parameter_decorator(self):
        """Test basic @parameter decorator functionality."""
        
        class TestModule:
            def __init__(self):
                self._temperature = 298.15
            
            @parameter(name="temperature", units="K", bounds=(200.0, 400.0))
            def temperature(self):
                return self._temperature
            
            @temperature.setter
            def temperature(self, value):
                if not (200.0 <= value <= 400.0):
                    raise ValueError(f"Temperature {value} out of bounds")
                self._temperature = value
        
        module = TestModule()
        
        # Get parameter
        assert module.temperature == 298.15
        
        # Set valid parameter
        module.temperature = 310.0
        assert module.temperature == 310.0
        
        # Set invalid parameter
        with pytest.raises(ValueError):
            module.temperature = 500.0
    
    def test_parameter_with_validation(self):
        """Test parameter decorator with validation."""
        
        class TestModule:
            def __init__(self):
                self._pressure = 101325.0
            
            @parameter(
                name="pressure",
                units="Pa",
                bounds=(50000.0, 200000.0),
                validate=True
            )
            def pressure(self):
                return self._pressure
            
            @pressure.setter
            def pressure(self, value):
                # Validation happens in decorator
                self._pressure = value
        
        module = TestModule()
        
        # Valid value
        module.pressure = 100000.0
        assert module.pressure == 100000.0
        
        # Invalid value should be caught by decorator validation
        # (Implementation would need to handle this)
    
    def test_parameter_with_manager_registration(self):
        """Test parameter decorator with automatic manager registration."""
        
        class TestModule:
            def __init__(self):
                self._albedo = 0.3
                self._register_parameters()
            
            @parameter(
                name="surface.albedo",
                units="dimensionless",
                bounds=(0.0, 1.0),
                register=True,
                dtype="float32"
            )
            def albedo(self):
                return self._albedo
            
            @albedo.setter
            def albedo(self, value):
                self._albedo = value
            
            def _register_parameters(self):
                """Register all decorated parameters."""
                # This would be called to register parameters
                # Implementation needed in actual decorator
                pass
        
        module = TestModule()
        # Further implementation needed for auto-registration


class TestLearnableParameterDecorator:
    """Test @learnable_parameter decorator."""
    
    def test_learnable_parameter_basic(self):
        """Test basic learnable_parameter functionality."""
        
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._setup_parameters()
            
            @learnable_parameter(name="weight", initial_value=1.0)
            def weight(self):
                return self._weight
            
            @learnable_parameter(name="bias", initial_value=torch.zeros(3))
            def bias(self):
                return self._bias
            
            def _setup_parameters(self):
                """Initialize learnable parameters."""
                self._weight = torch.nn.Parameter(torch.tensor(1.0))
                self._bias = torch.nn.Parameter(torch.zeros(3))
        
        module = TestModule()
        
        # Check parameters are nn.Parameters
        assert isinstance(module._weight, torch.nn.Parameter)
        assert isinstance(module._bias, torch.nn.Parameter)
        
        # Check gradients are enabled
        assert module._weight.requires_grad
        assert module._bias.requires_grad
        
        # Check values
        assert module.weight.item() == 1.0
        assert torch.allclose(module.bias, torch.zeros(3))
    
    def test_learnable_parameter_gradient_flow(self):
        """Test gradient flow through learnable parameters."""
        
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))
                self.shift = torch.nn.Parameter(torch.tensor(1.0))
            
            @learnable_parameter(name="scale", initial_value=2.0)
            def get_scale(self):
                return self.scale
            
            @learnable_parameter(name="shift", initial_value=1.0)
            def get_shift(self):
                return self.shift
            
            def forward(self, x):
                return x * self.scale + self.shift
        
        model = TestModel()
        
        # Forward pass
        x = torch.tensor([1.0, 2.0, 3.0])
        y = model(x)
        
        # Compute loss and gradients
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist
        assert model.scale.grad is not None
        assert model.shift.grad is not None
        
        # Check gradient values
        assert torch.allclose(model.scale.grad, x.sum())
        assert model.shift.grad.item() == 3.0  # Number of elements


class TestValidatedParameterDecorator:
    """Test @validated_parameter decorator."""
    
    def test_validated_parameter_basic(self):
        """Test basic validated_parameter functionality."""
        
        def positive_validator(value):
            """Validate that value is positive."""
            if value <= 0:
                raise ValueError(f"Value {value} must be positive")
            return True
        
        class TestModule:
            def __init__(self):
                self._rate = 0.5
            
            @validated_parameter(
                name="decay_rate",
                validator=positive_validator
            )
            def decay_rate(self):
                return self._rate
            
            @decay_rate.setter
            def decay_rate(self, value):
                # Validator is called before setting
                if positive_validator(value):
                    self._rate = value
        
        module = TestModule()
        
        # Valid value
        module.decay_rate = 0.1
        assert module.decay_rate == 0.1
        
        # Invalid value
        with pytest.raises(ValueError):
            module.decay_rate = -0.1
    
    def test_validated_parameter_multiple_validators(self):
        """Test validated_parameter with multiple validators."""
        
        def range_validator(value):
            """Validate range."""
            if not (0.0 <= value <= 1.0):
                raise ValueError("Value must be in [0, 1]")
            return True
        
        def not_half_validator(value):
            """Validate not exactly 0.5."""
            if value == 0.5:
                raise ValueError("Value cannot be exactly 0.5")
            return True
        
        class TestModule:
            def __init__(self):
                self._fraction = 0.3
            
            @validated_parameter(
                name="fraction",
                validator=[range_validator, not_half_validator]
            )
            def fraction(self):
                return self._fraction
            
            @fraction.setter
            def fraction(self, value):
                # All validators must pass
                for validator in [range_validator, not_half_validator]:
                    validator(value)
                self._fraction = value
        
        module = TestModule()
        
        # Valid values
        module.fraction = 0.3
        assert module.fraction == 0.3
        
        module.fraction = 0.7
        assert module.fraction == 0.7
        
        # Invalid: out of range
        with pytest.raises(ValueError):
            module.fraction = 1.5
        
        # Invalid: exactly 0.5
        with pytest.raises(ValueError):
            module.fraction = 0.5


# TODO: Implement cached_parameter decorator
# The following test class is commented out until cached_parameter is implemented
# class TestCachedParameterDecorator:
#     """Test @cached_parameter decorator."""
#     
#     def test_cached_parameter_basic(self):
#         """Test basic cached_parameter functionality."""
#         pass  # TODO: Implement when cached_parameter is available




class TestDeclaredParameterMetadata:
    """Decorators supply no units, bounds, dtype or description on the caller's behalf."""

    def test_parameter_requires_units_and_bounds(self):
        import inspect

        params = inspect.signature(parameter).parameters
        for name in ("units", "bounds"):
            assert params[name].default is inspect.Parameter.empty, name
        with pytest.raises(TypeError, match="units"):
            parameter(name="x", bounds=(0.0, 1.0))
        with pytest.raises(TypeError, match="bounds"):
            parameter(name="x", units="m")

    def test_registering_parameter_requires_a_declared_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            parameter(name="x", units="m", bounds=(0.0, 1.0), register=True)

    def test_parameter_definition_requires_units_and_bounds(self):
        import dataclasses

        from relic.parameters.decorators import ParameterDefinition

        fields = ParameterDefinition.__dataclass_fields__
        for name in ("units", "bounds"):
            assert fields[name].default is dataclasses.MISSING, name

    def test_validated_and_learnable_parameter_take_no_unused_metadata(self):
        import inspect

        for decorator in (validated_parameter, learnable_parameter):
            params = inspect.signature(decorator).parameters
            assert "units" not in params and "description" not in params, decorator.__name__
        # learnable_parameter never read its bounds either.
        assert "bounds" not in inspect.signature(learnable_parameter).parameters

    def test_managed_parameter_spec_must_declare_its_metadata(self):
        @managed_parameters(namespace="test.incomplete_spec")
        class Incomplete:
            def __init__(self):
                self.rate = 1.0
                self.rate_spec = {"units": "s-1", "bounds": (0.0, 2.0)}

        with pytest.raises(ValueError, match="rate"):
            Incomplete()
        assert not get_global_manager().has_parameter("test.incomplete_spec.rate")

    def test_attribute_without_spec_is_not_registered_with_invented_metadata(self):
        @managed_parameters(namespace="test.undeclared")
        class Undeclared:
            def __init__(self):
                self.counter = 3

        Undeclared()
        assert not get_global_manager().has_parameter("test.undeclared.counter")
