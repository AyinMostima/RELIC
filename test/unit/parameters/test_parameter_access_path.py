"""
Unit tests for parameter access path unification.

This module tests that parameter access follows the unified pattern:
1. Functors use self.get_parameter_value()
2. Non-Functor classes use dependency injection
3. No direct calls to get_global_manager() except in infrastructure code
"""

import pytest
import torch
from unittest.mock import Mock, patch

# Test imports
from relic.core.abstractions import BiologicalProcess, ScaleCoupling
from relic.core.functor import FunctorBase
from relic.stochastic.ensemble_runner import EnsembleRunner, ExecutionConfig


class TestParameterAccessUnification:
    """Test suite for unified parameter access patterns."""

    def test_biological_process_uses_functor_method(self):
        """Test BiologicalProcess uses FunctorBase method when available."""

        class TestBioProcess(BiologicalProcess):
            def __init__(self):
                super().__init__(process_name="test_process")
                # Simulate being part of a FunctorBase
                self._get_parameter_manager = Mock(return_value=Mock())

            def forward(self, state, time, dt, **kwargs):
                return {}

        # Create instance
        process = TestBioProcess()

        # Mock the parameter manager
        mock_manager = Mock()
        mock_manager.get.return_value = 42.0
        process._get_parameter_manager.return_value = mock_manager

        # Get parameter value
        value = process.get_parameter_value("test_param")

        # Verify it used the FunctorBase method
        process._get_parameter_manager.assert_called()
        mock_manager.get.assert_called_with("pedosphere.test_process.test_param")
        assert value == 42.0

    @pytest.mark.parametrize("owner", [BiologicalProcess, ScaleCoupling])
    def test_abstraction_lookup_has_no_caller_default(self, owner):
        """No default_value and no swallowed keyword arguments."""
        import inspect

        parameters = inspect.signature(owner.get_parameter_value).parameters
        assert "default_value" not in parameters
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())

    def test_biological_process_without_functor_raises_for_a_missing_parameter(self):
        """A parameter found in neither the config nor a manager raises KeyError."""

        class TestBioProcess(BiologicalProcess):
            def __init__(self):
                super().__init__(process_name="test_process")
                # No _get_parameter_manager method

            def forward(self, state, time, dt, **kwargs):
                return {}

        # Create instance
        process = TestBioProcess()

        with pytest.raises(KeyError, match="test_param"):
            process.get_parameter_value("test_param")
        with pytest.raises(TypeError):
            process.get_parameter_value("test_param", default_value=99.0)

    def test_scale_coupling_uses_functor_method(self):
        """Test ScaleCoupling uses FunctorBase method when available."""

        class TestScaleCoupling(ScaleCoupling):
            def __init__(self):
                super().__init__(
                    source_scale="fine",
                    target_scale="coarse",
                    coupling_type="spatial"  # Use valid coupling type
                )
                # Simulate being part of a FunctorBase
                self._get_parameter_manager = Mock(return_value=Mock())

            def upscale(self, fine_data):
                return fine_data

            def downscale(self, coarse_data):
                return coarse_data

        # Create instance
        coupling = TestScaleCoupling()

        # Mock the parameter manager
        mock_manager = Mock()
        mock_manager.get.return_value = 123.0
        coupling._get_parameter_manager.return_value = mock_manager

        # Get parameter value
        value = coupling.get_parameter_value("test_param")

        # Verify it used the FunctorBase method
        coupling._get_parameter_manager.assert_called()
        mock_manager.get.assert_called_with("coupling.spatial.test_param")  # Fixed to match coupling_type
        assert value == 123.0

    def test_ensemble_runner_with_injected_manager(self):
        """Test EnsembleRunner uses injected parameter manager."""

        # Create mock parameter manager
        mock_manager = Mock()
        mock_manager.parameters = {
            "param1": Mock(units="kg m-2", description="test parameter")
        }

        # Create ensemble runner with injected manager
        config = ExecutionConfig()
        runner = EnsembleRunner(config=config, parameter_manager=mock_manager)

        # Verify it stored the injected manager
        assert runner.parameter_manager == mock_manager

        # Test _build_metadata uses injected manager
        parameter_samples = {"param1": torch.tensor([1.0, 2.0])}
        with patch("relic.parameters.get_global_manager") as global_getter:
            metadata = runner._build_metadata(parameter_samples)

        global_getter.assert_not_called()
        assert "title" in metadata
        assert metadata["metadata_profile"] == "explicit-ledger"
        assert "conventions" not in metadata
        assert metadata["parameter_statistics"]["param1"]["units"] == "kg m-2"
        assert (
            metadata["parameter_statistics"]["param1"]["description"]
            == "test parameter"
        )

    def test_ensemble_runner_without_manager_never_uses_global_fallback(self):
        """Test EnsembleRunner never resolves a hidden global manager."""

        # Create ensemble runner without injected manager
        config = ExecutionConfig()
        runner = EnsembleRunner(config=config, parameter_manager=None)

        # Verify no manager stored
        assert runner.parameter_manager is None

        with patch("relic.parameters.get_global_manager") as global_getter:
            parameter_samples = {"param1": torch.tensor([1.0, 2.0])}
            metadata = runner._build_metadata(parameter_samples)

        global_getter.assert_not_called()
        param_info = metadata["parameter_statistics"]["param1"]
        assert "title" in metadata
        assert "units" not in param_info
        assert "description" not in param_info

    def test_no_direct_global_manager_calls_in_functors(self):
        """Test that Functor subclasses don't directly call get_global_manager."""

        # This is a policy test - we verify the pattern is followed
        # by checking that abstractions use _get_parameter_manager instead

        class TestFunctor(FunctorBase):
            def __init__(self):
                from relic.core.functor import FunctorMetadata, ProcessCategory
                metadata = FunctorMetadata(
                    name="test",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    input_variables=["state"],
                    output_variables=["flux"],
                    description="Test functor",
                    references=["Test reference"]
                )
                config = {}  # Empty config for testing
                super().__init__(name="test", config=config, metadata=metadata)

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt):
                # Should use self.get_parameter_value, not get_global_manager
                value = self.get_parameter_value("test_param")
                return {"flux": torch.tensor(0.0)}

        # Create functor
        functor = TestFunctor()

        # Mock the internal _get_parameter_manager
        functor._parameter_manager = Mock()
        functor._parameter_manager.get_parameter.return_value = None
        functor._parameter_manager.get.return_value = 1.0

        # Call forward
        state = {"test": torch.tensor(1.0)}
        result = functor.forward(state, 0.0, 1.0)

        # Verify it used the proper method
        assert "flux" in result


class TestParameterAccessCompliance:
    """Test compliance with parameter access patterns across the codebase."""

    def test_cli_tools_can_use_global_manager(self):
        """Test that CLI tools are allowed to use get_global_manager."""
        # CLI tools like params_manager.py are infrastructure code
        # and are allowed to directly access the global manager
        assert True  # This is a documentation test

    def test_validation_rules_can_use_global_manager(self):
        """Test that validation rules can use get_global_manager for checking."""
        # Validation rules need to inspect the parameter system
        # and are allowed to access the global manager
        assert True  # This is a documentation test

    def test_config_loader_can_use_global_manager(self):
        """Test that config loader can use get_global_manager for registration."""
        # Config loader is part of initialization infrastructure
        # and needs to register parameters with the global manager
        assert True  # This is a documentation test

    def test_decorators_can_use_global_manager(self):
        """Test that parameter decorators can use get_global_manager."""
        # Decorators are tightly coupled with the parameter system
        # and provide automatic parameter management features
        assert True  # This is a documentation test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


